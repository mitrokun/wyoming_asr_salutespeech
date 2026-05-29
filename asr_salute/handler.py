import asyncio
import logging
from typing import AsyncGenerator

import grpc
from wyoming.asr import Transcript, TranscriptStart, TranscriptStop, TranscriptChunk
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .api import recognition_pb2, recognition_pb2_grpc
from .auth import SaluteSpeechAuth

_LOGGER = logging.getLogger(__name__)

class SberEventHandler(AsyncEventHandler):
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        auth: SaluteSpeechAuth,
        ca_cert_content: bytes,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.auth = auth
        self.ca_cert_content = ca_cert_content
        self.audio_queue = asyncio.Queue()
        self.recognition_task = None
        self.is_streaming = False
        self.full_transcript = []
        self.sent_words = []
        self.session_has_text = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        if AudioStart.is_type(event.type):
            _LOGGER.debug("AudioStart: Starting recognition session")
            self.is_streaming = True
            self.full_transcript = []
            self.sent_words = []
            self.session_has_text = False
            
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                
            self.recognition_task = asyncio.create_task(self._run_recognition())
            return True

        if AudioChunk.is_type(event.type):
            if self.is_streaming:
                chunk = AudioChunk.from_event(event)
                await self.audio_queue.put(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("AudioStop: Client finished sending audio")
            self.is_streaming = False
            await self.audio_queue.put(None)
            
            if self.recognition_task:
                await self.recognition_task
            return False

        return True

    async def _request_generator(self, options) -> AsyncGenerator[recognition_pb2.RecognitionRequest, None]:
        yield recognition_pb2.RecognitionRequest(options=options)
        while True:
            audio_data = await self.audio_queue.get()
            if audio_data is None:
                break
            yield recognition_pb2.RecognitionRequest(audio_chunk=audio_data)

    async def _run_recognition(self):
        try:
            token = await self.auth.get_access_token()
            
            ssl_cred = grpc.ssl_channel_credentials(root_certificates=self.ca_cert_content)
            token_cred = grpc.access_token_call_credentials(token)
            composite_creds = grpc.composite_channel_credentials(ssl_cred, token_cred)

            options = recognition_pb2.RecognitionOptions()
            options.audio_encoding = recognition_pb2.RecognitionOptions.PCM_S16LE
            options.sample_rate = 16000
            options.channels_count = 1
            
            options.enable_multi_utterance.enable = True 
            options.enable_partial_results.enable = True 

            async with grpc.aio.secure_channel("smartspeech.sber.ru:443", composite_creds) as channel:
                stub = recognition_pb2_grpc.SmartSpeechStub(channel)
                
                await self.write_event(TranscriptStart().event())

                stream = stub.Recognize(self._request_generator(options))
                
                self.sent_words = []
                self.session_has_text = False

                async for response in stream:
                    if response.HasField("transcription"):
                        tr = response.transcription
                        
                        current_sentence = " ".join([hyp.normalized_text for hyp in tr.results]).strip()
                        
                        if current_sentence:
                            current_words = current_sentence.split()
                            deltas_to_send = []
                            
                            for i, new_word in enumerate(current_words):
                                # Кейс 1: Старое завершенное слово (пропускаем без изменений для сохранения индексации)
                                if i < len(self.sent_words) - 1:
                                    continue
                                
                                # Кейс 2: Слово на стыке (которое могло быть неполным, например "по" -> "попробуем")
                                elif i == len(self.sent_words) - 1:
                                    old_word = self.sent_words[i]
                                    if new_word == old_word:
                                        continue
                                    elif new_word.startswith(old_word):
                                        # Это продолжение слова, берем только суффикс без пробела
                                        suffix = new_word[len(old_word):]
                                        deltas_to_send.append(suffix)
                                        self.sent_words[i] = new_word
                                    else:
                                        # Слово изменилось полностью (например, "мы" -> "вы").
                                        # Игнорируем замену на клиенте ради сохранения стабильности верстки,
                                        # но сохраняем индекс выравнивания.
                                        continue
                                
                                # Кейс 3: Новое слово (индекс больше длины отправленных слов)
                                else:
                                    if self.session_has_text:
                                        delta_word = " " + new_word
                                    else:
                                        delta_word = new_word
                                        self.session_has_text = True
                                        
                                    deltas_to_send.append(delta_word)
                                    self.sent_words.append(new_word)
                            
                            if deltas_to_send:
                                final_delta = "".join(deltas_to_send)
                                await self.write_event(TranscriptChunk(text=final_delta).event())
                        
                        if tr.eou:
                            if current_sentence:
                                _LOGGER.debug("Utterance recognized: %s", current_sentence)
                                self.full_transcript.append(current_sentence)
                            
                            self.sent_words = []
                               
            final_text = " ".join(self.full_transcript)
            
            if final_text:
                _LOGGER.info("Final Full Text: %s", final_text)
                await self.write_event(Transcript(text=final_text).event())
            
            await self.write_event(TranscriptStop().event())

        except grpc.RpcError as e:
            _LOGGER.warning("gRPC Error: %s", e)
        except Exception as e:
            _LOGGER.exception("Recognition pipeline failed")
        finally:
            self.recognition_task = None
