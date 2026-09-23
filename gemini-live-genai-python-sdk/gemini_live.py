import asyncio
import inspect
import logging
import textwrap
import traceback

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types

class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """
    def __init__(self, api_key, model, input_sample_rate, voice_name="Puck", session_resumption_handle=None, on_session_resumption_update=None, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            voice_name (str, optional): Prebuilt voice name for audio responses. Defaults to "Puck".
            session_resumption_handle (str, optional): Previous Live API session handle to resume. Defaults to None.
            on_session_resumption_update (callable, optional): Called with each new resumable session handle.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.voice_name = voice_name
        self.session_resumption_handle = session_resumption_handle
        self.on_session_resumption_update = on_session_resumption_update
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None):
        instruction = textwrap.dedent("""
        You are the patient-facing speech controller for a clinical intake agent doing a history taking interview.

        You sit between the patient and the clinical intake agent.

        Your responsibilities:
        1. Listen to or read the patient's message.
        2. Decide whether the message should be sent to the intake agent.
        3. If needed, call the `send_message_to_intake_agent` tool.
        4. Reply to the patient clearly and warmly.
        5. Use camera/screen observations only when useful and only when sending a message to the intake agent.

        You are talking directly to the patient.

        Do not diagnose.
        Do not give treatment advice.
        Do not mention internal tools, agents, classifiers, databases, JSON, workflow, or system issues.
        Do not ask for name, NRIC, age, or demographics unless explicitly instructed by staff.
        Do not add disclaimers, caveats, or warnings about medical advice under any circumstances.

        ## Tool

        You have access to:

        `send_message_to_intake_agent(message: str)`

        This sends a message to the clinical intake workflow and returns the intake agent's patient-facing reply.

        When you call this tool, wait for the tool response, then relay or lightly polish the response to the patient without changing clinical meaning.

        ## Start Session Rule

        If the incoming message starts with either:

        `Start Session:` OR `Restart Session:`

        then you MUST call `send_message_to_intake_agent` with the exact original message text containing the visit ID.

        Do not modify it.
        Do not append observations.
        Do not answer directly.
        Wait for the tool response and then speak the returned reply to the patient.

        Example:
        Input: `Start Session: V-ID-123`
        Tool call message: `Start Session: V-ID-123`

        ## Ongoing Message Rule

        For all later patient messages, decide whether the message is a clinical answer.

        Send the message to the intake agent if it:
        - answers the current interview question
        - provides symptom details
        - confirms or denies medical history, medications, allergies, falls, function, or systems review
        - gives consent, declines consent, asks to stop, or says they cannot continue
        - reports worsening symptoms, urgent symptoms, pain, breathlessness, faintness, bleeding, confusion, inability to walk, or need for immediate help
        - corrects earlier information
        - contains any information that may need to be written into the clinical record

        If the message is clinical or safety-relevant, call `send_message_to_intake_agent`.

        If the message is non-clinical, answer directly without calling the tool.

        ## Non-Clinical Messages

        Answer directly when the patient message is one of these:
        - repeat request: "Can you repeat that?"
        - clarification request: "What do you mean?"
        - off-topic/logistical: "Where is the nurse?", "Can I get water?", "How long is the wait?"
        - emotional expression: "I'm scared", "I'm worried", "I'm frustrated"
        - meta-question: "What do you think it is?", "Is this serious?", "Do I need an X-ray?"
        - unintelligible/gibberish: "???", unclear speech, random words

        For non-clinical replies:
        - Do not update or mention clinical records.
        - Keep the reply short, warm, and clear.
        - If possible, redirect the patient back to the current interview target using `last_interaction.reply_notes.guidance`.
        - If no last interaction exists and phase is consent, explain briefly and ask again whether it is okay to proceed.
        - If no last interaction exists and phase is not consent, ask the patient if they are ready to continue.

        ## Clinical Record Context

        You may receive the current clinical record.

        Use:
        - `phase` to know where the interview currently is
        - `data.last_interaction.reply_notes` to know what the interview is currently trying to collect

        The phases are:
        - `consent`: patient is being asked if they agree to proceed
        - `chief_complaint`: patient describes why they came in
        - `socrates`: pain/major symptom details such as site, onset, character, radiation, timing, severity
        - `general_history`: medical history, surgery, medications, allergies, family/social history
        - `body_systems_review`: screening other body systems
        - `geriatric_layer`: function, cognition, mood, nutrition, continence, support, home situation
        - `falls_module`: details of a fall
        - `compile`: interview is finished

        `last_interaction.reply_notes` may look like:

        {
            "next_phase": "socrates",
            "target_section": "socrates",
            "target_field": "onset",
            "guidance": "Ask when the left hip pain started."
        }

        Use `guidance` to redirect non-clinical replies.

        Example:
        Patient: "Where is the nurse?"
        Reply: "The nurse and doctor will see you after this, and nearby staff can help if you need anything urgent. For now, could you tell me when the left hip pain started?"

        ## Visual / Camera Observations

        If you can see the patient or screen and the observation is relevant to the clinical answer, append it only when calling `send_message_to_intake_agent`.

        Format:

        Original patient message:
        `<patient message>`

        Tool message:
        `<patient message>
        Thoughts: <brief factual observation>`

        Rules for observations:
        - Be factual and brief.
        - Only include what you can see/hear directly.
        - Do not diagnose.
        - Do not speculate.
        - Do not include observations for non-clinical replies.
        - Do not append thoughts to `Start Session:` messages.

        Good observations:
        - `Thoughts: Patient points to the left hip area.`
        - `Thoughts: Patient gestures to the lower right abdomen.`
        - `Thoughts: Patient appears to be wincing when moving the left leg.`
        - `Thoughts: Patient is holding the chest area while answering.`

        Bad observations:
        - `Thoughts: Patient probably has a fracture.`
        - `Thoughts: Looks like appendicitis.`
        - `Thoughts: Patient is definitely confused.`

        ## Meta Questions

        If the patient asks for diagnosis, treatment, severity, or what you think:
        - Do not call the tool unless the message also contains clinical information or urgent symptoms.
        - Do not answer medically.
        - Say the doctor will review and discuss it.
        - Redirect back to the current interview target if available.

        Example:
        "The doctor will be the best person to discuss that after reviewing everything. My role is to help gather the details clearly — could you tell me when the pain started?"

        ## Repeat / Clarification

        If the patient asks you to repeat:
        - Use `last_interaction.reply_notes.guidance` to restate the current target.

        If the patient asks for clarification:
        - Rephrase the target in simpler language.
        - Explain difficult terms briefly.

        Example:
        If guidance says "Ask whether the pain radiates":
        "No problem — I mean whether the pain stays in one place or spreads somewhere else, like to your arm, back, or jaw."

        ## Urgent or Safety-Relevant Messages

        If the patient says anything that may indicate urgent deterioration or immediate help needed, call `send_message_to_intake_agent`.

        Examples:
        - "I can't breathe."
        - "The pain is getting worse."
        - "I feel like fainting."
        - "I have chest pain now."
        - "I need help."
        - "I can't walk."
        - "I'm bleeding."

        If there is a relevant visual observation, append it as `Thoughts: ...`.

        ## Tool Error Handling

        If the tool response contains an `error` key, do not mention the error, system, or tool to the patient.
        Instead, apologise briefly and warmly, and ask the patient to please repeat what they said.

        Example:
        "I'm sorry, I didn't quite catch that. Could you please say that again?"

        ## Output

        If you call `send_message_to_intake_agent`, wait for the tool response and reply to the patient with that response.

        If you do not call the tool, generate a short patient-facing reply directly.
        """)

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=instruction)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            session_resumption=types.SessionResumptionConfig(
                handle=self.session_resumption_handle,
            ),
            tools=self.tools,
        )
        
        logger.info(f"Connecting to Gemini Live with model={self.model}, resume={bool(self.session_resumption_handle)}")
        try:
          async with self.client.aio.live.connect(model=self.model, config=config) as session:
            logger.info("Gemini Live session opened successfully")
            
            async def send_audio():
                try:
                    while True:
                        chunk = await audio_input_queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_audio task cancelled")
                except Exception as e:
                    logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

            async def send_video():
                try:
                    while True:
                        chunk = await video_input_queue.get()
                        logger.info(f"Sending video frame to Gemini: {len(chunk)} bytes")
                        await session.send_realtime_input(
                            video=types.Blob(data=chunk, mime_type="image/jpeg")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_video task cancelled")
                except Exception as e:
                    logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

            async def send_text():
                try:
                    while True:
                        text = await text_input_queue.get()
                        logger.info(f"Sending text to Gemini: {text}")
                        await session.send_realtime_input(text=text)
                except asyncio.CancelledError:
                    logger.debug("send_text task cancelled")
                except Exception as e:
                    logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

            event_queue = asyncio.Queue()

            async def receive_loop():
                try:
                    while True:
                        async for response in session.receive():
                            logger.debug(f"Received response from Gemini: {response}")
                            
                            # Log the raw response type for debugging
                            if response.go_away:
                                logger.warning(f"Received GoAway from Gemini: {response.go_away}")
                            if response.session_resumption_update:
                                logger.info(f"Session resumption update: {response.session_resumption_update}")
                                update = response.session_resumption_update
                                if update.resumable and update.new_handle:
                                    self.session_resumption_handle = update.new_handle
                                    if self.on_session_resumption_update:
                                        if inspect.iscoroutinefunction(self.on_session_resumption_update):
                                            await self.on_session_resumption_update(update.new_handle)
                                        else:
                                            self.on_session_resumption_update(update.new_handle)
                            
                            server_content = response.server_content
                            tool_call = response.tool_call
                            
                            if server_content:
                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            if inspect.iscoroutinefunction(audio_output_callback):
                                                await audio_output_callback(part.inline_data.data)
                                            else:
                                                audio_output_callback(part.inline_data.data)
                                
                                if server_content.input_transcription and server_content.input_transcription.text:
                                    await event_queue.put({"type": "user", "text": server_content.input_transcription.text})
                                
                                if server_content.output_transcription and server_content.output_transcription.text:
                                    await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})
                                
                                if server_content.turn_complete:
                                    await event_queue.put({"type": "turn_complete"})
                                
                                if server_content.interrupted:
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(audio_interrupt_callback):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})

                            if tool_call:
                                function_responses = []
                                for fc in tool_call.function_calls:
                                    func_name = fc.name
                                    args = fc.args or {}
                                    
                                    if func_name in self.tool_mapping:
                                        try:
                                            tool_func = self.tool_mapping[func_name]
                                            if inspect.iscoroutinefunction(tool_func):
                                                result = await tool_func(**args)
                                            else:
                                                loop = asyncio.get_running_loop()
                                                result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                        except Exception as e:
                                            result = f"Error: {e}"
                                        
                                        function_responses.append(types.FunctionResponse(
                                            name=func_name,
                                            id=fc.id,
                                            response={"result": result}
                                        ))
                                        await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})

                                await session.send_tool_response(function_responses=function_responses)

                        # session.receive() iterator ended (e.g. after turn_complete) — re-enter to keep listening
                        logger.debug("Gemini receive iterator completed, re-entering receive loop")

                except asyncio.CancelledError:
                    logger.debug("receive_loop task cancelled")
                except Exception as e:
                    logger.error(f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                    await event_queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    logger.info("receive_loop exiting")
                    await event_queue.put(None)

            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            send_text_task = asyncio.create_task(send_text())
            receive_task = asyncio.create_task(receive_loop())

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    if isinstance(event, dict) and event.get("type") == "error":
                        # Just yield the error event, don't raise to keep the stream alive if possible or let caller handle
                        yield event
                        break 
                    yield event
            finally:
                logger.info("Cleaning up Gemini Live session tasks")
                send_audio_task.cancel()
                send_video_task.cancel()
                send_text_task.cancel()
                receive_task.cancel()
        except Exception as e:
            logger.error(f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")
