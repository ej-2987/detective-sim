
import { GoogleGenAI, HarmCategory, HarmBlockThreshold, Chat, GenerateContentResponse } from "@google/genai";
import type { CaseDetails, CharacterRole, Message } from '../types';
import { GEMINI_TEXT_MODEL, PERSONA_PROMPTS, INITIAL_SYSTEM_INSTRUCTIONS } from '../constants';

const parseJsonFromMarkdown = (text: string): any => {
  let jsonStr = text.trim();
  const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
  const match = jsonStr.match(fenceRegex);
  if (match && match[2]) {
    jsonStr = match[2].trim();
  }
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    console.error("Failed to parse JSON response:", e, "Original text:", text);
    // Return a structured error or re-throw
    throw new Error(`Invalid JSON response from AI. Could not parse: ${text.substring(0,100)}...`);
  }
};

const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

const getGoogleGenAIInstance = (apiKey: string) => {
    if (!apiKey) {
        throw new Error("API Key is not provided to GoogleGenAI constructor.");
    }
    return new GoogleGenAI({ apiKey });
}

export const generateCaseDetails = async (apiKey: string, precedentText: string): Promise<CaseDetails> => {
  const ai = getGoogleGenAIInstance(apiKey);
  const model = GEMINI_TEXT_MODEL;
  
  const prompt = `다음 판례를 기반으로 사건 개요, 중요 쟁점 사항 (3-5개), 그리고 기본적인 수사 계획을 생성해주세요. 결과를 다음 JSON 형식으로 반환해주세요: {"overview": "...", "issues": ["...", "..."], "plan": ["...", "..."]}\n\n판례:\n${precedentText}`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
        model: model,
        contents: prompt,
        config: { 
            responseMimeType: "application/json",
            temperature: 0.5 
        },
        // safetySettings, // As per previous instruction, safetySettings were removed. If needed, re-add.
    });
    
    const rawJson = response.text;
    const parsedDetails = parseJsonFromMarkdown(rawJson);
    // Basic validation for the expected structure
    if (!parsedDetails.overview || !Array.isArray(parsedDetails.issues) || !Array.isArray(parsedDetails.plan)) {
        throw new Error("AI response for case details is missing required fields (overview, issues, plan).");
    }
    return parsedDetails as CaseDetails;

  } catch (error) {
    console.error("Error generating case details:", error);
    if (error instanceof Error && error.message.includes("API Key not valid")) {
        throw new Error("API 키가 유효하지 않습니다. 확인 후 다시 시도해주세요.");
    }
    throw new Error(`Gemini로부터 사건 상세 정보를 생성하는 데 실패했습니다. ${error instanceof Error ? error.message : String(error)}`);
  }
};

export const generateComplaint = async (apiKey: string, caseOverview: string): Promise<string> => {
  const ai = getGoogleGenAIInstance(apiKey);
  const model = GEMINI_TEXT_MODEL;

  const prompt = `다음 사건 개요를 바탕으로, 감정적이고 비논리적인 말투로 고소장을 작성해주세요. 고소 내용은 상세하지만, 반드시 일관적일 필요는 없습니다. 불의에 대한 감정과 피고소인에 대한 분노를 강조하고, 형식적인 법률 용어는 사용하지 마세요.\n\n사건 개요:\n${caseOverview}`;
  
  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
        model: model,
        contents: prompt,
        config: { temperature: 0.8 },
        // safetySettings,
    });
    if (!response.text) {
        throw new Error("AI가 고소장 내용 생성에 실패했습니다 (빈 응답).");
    }
    return response.text;
  } catch (error) {
    console.error("Error generating complaint:", error);
    if (error instanceof Error && error.message.includes("API Key not valid")) {
        throw new Error("API 키가 유효하지 않습니다. 확인 후 다시 시도해주세요.");
    }
    throw new Error(`Gemini로부터 고소장을 생성하는 데 실패했습니다. ${error instanceof Error ? error.message : String(error)}`);
  }
};

export const sendMessageToCharacter = async (
  apiKey: string,
  characterRole: CharacterRole,
  message: string,
  caseOverview: string,
  previousMessages: Message[], // Full history of this character including current user message
  existingChatInstance: Chat | null
): Promise<{response: string; updatedChat: Chat}> => {
  const ai = getGoogleGenAIInstance(apiKey);
  const model = GEMINI_TEXT_MODEL;

  // Reconstruct history for the Chat object, excluding the latest user message which will be sent via sendMessage
  const chatHistoryForGemini = previousMessages.slice(0, -1).map(msg => ({
    role: msg.sender === 'user' ? 'user' : 'model',
    parts: [{ text: msg.text }]
  }));

  let chat: Chat;
  if (existingChatInstance) {
    chat = existingChatInstance;
    // Note: The history of existingChatInstance might need careful management 
    // if it's intended to be truly stateful across multiple sendMessage calls
    // with complex history alterations. For this app's flow, re-creating or 
    // ensuring history matches `previousMessages` might be safer if issues arise.
  } else {
     chat = ai.chats.create({
        model: model,
        config: { 
            systemInstruction: INITIAL_SYSTEM_INSTRUCTIONS[characterRole].replace('{CASE_OVERVIEW}', caseOverview),
            temperature: 0.75, // Slightly higher temp for more varied responses
            // safetySettings,
        },
        history: chatHistoryForGemini 
    });
  }
  
  // The user's current message (last in previousMessages) is what we're sending now.
  const currentUserMessage = previousMessages[previousMessages.length -1].text;

  // For robust persona and context, especially if chat object isn't perfectly stateful
  // or if we want to ensure full context on each turn:
  const previousStatementsFormatted = previousMessages
    .slice(0, -1) // Exclude current user message being sent
    .map(msg => `${msg.sender === 'user' ? '수사관' : characterRole}: ${msg.text}`)
    .join('\n');
  
  const fullPromptForThisTurn = `${PERSONA_PROMPTS[characterRole]
    .replace('{CASE_OVERVIEW}', caseOverview)
    .replace('{PREVIOUS_STATEMENTS}', previousStatementsFormatted || "이전 진술 없음.")
  }\n\n수사관: ${currentUserMessage}`;


  try {
    // Send the user's current message. The persona and history are reinforced by the system prompt logic.
    const result: GenerateContentResponse = await chat.sendMessage({ message: fullPromptForThisTurn });
    
    if (!result.text) {
        throw new Error("AI가 캐릭터 응답 생성에 실패했습니다 (빈 응답).");
    }
    return { response: result.text, updatedChat: chat };
  } catch (error) {
    console.error(`Error sending message to ${characterRole}:`, error);
    if (error instanceof Error && error.message.includes("API Key not valid")) {
        throw new Error("API 키가 유효하지 않습니다. 확인 후 다시 시도해주세요.");
    }
    throw new Error(`${characterRole}에게 메시지 전송 실패: ${error instanceof Error ? error.message : String(error)}`);
  }
};

export const summarizeChatLog = async (apiKey: string, rawLog: string): Promise<string> => {
  const ai = getGoogleGenAIInstance(apiKey);
  const model = GEMINI_TEXT_MODEL;

  const prompt = `다음은 수사관과 조사 대상자 간의 대화 기록입니다. 이 대화 내용을 간결하고 전문적인 요약 로그로 정리해주세요. 주요 사실, 진술의 일관성 여부, 모순점, 감정적 반응 등을 중심으로 요약합니다.\n\n대화 기록:\n${rawLog}`;
  
  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
        model: model,
        contents: prompt,
        config: { temperature: 0.3 },
        // safetySettings, 
    });
    if (!response.text) {
        throw new Error("AI가 대화 요약에 실패했습니다 (빈 응답).");
    }
    return response.text;
  } catch (error) {
    console.error("Error summarizing chat log:", error);
    if (error instanceof Error && error.message.includes("API Key not valid")) {
        throw new Error("API 키가 유효하지 않습니다. 확인 후 다시 시도해주세요.");
    }
    throw new Error(`대화 요약 실패: ${error instanceof Error ? error.message : String(error)}`);
  }
};
