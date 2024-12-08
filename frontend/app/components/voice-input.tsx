"use client";

import { useState, useEffect, useRef } from "react";
import PatientHistory from "./patient-history";
import { duringSurgery } from "../actions/during-surgery";
import { FaMicrophone, FaStop } from "react-icons/fa";
import {
  SpeechRecognitionErrorEvent,
  SpeechRecognitionEvent,
} from "@/app/types/global";
import RoutingButtons from "./routing-buttons";
import FileUpload from "./file-upload";
import { uploadPreSurgeryFiles } from "../actions/pre-surgery-file-upload";

/**
 * VoiceInput Component
 * Captures user speech, sends it to the AI, and plays the AI's response as audio.
 */
const VoiceInput = () => {
  const recognitionRef = useRef<SpeechRecognition>();

  const [isActive, setIsActive] = useState<boolean>(false);
  const [surgeryProcedure, setSurgeryProcedure] = useState<boolean>(false);
  const [surgery, setSurgery] = useState<string>("");
  const [text, setText] = useState<string>("");
  const [patientHistory, setPatientHistory] = useState<string>("");
  const [aiResponse, setAIResponse] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const [files, setFiles] = useState<Record<string, File | null>>({
    prescription: null,
    scan: null,
    labReport: null,
  });
  const [uploadStatus, setUploadStatus] = useState("");

  const handleOperationChange = (
    event: React.ChangeEvent<HTMLInputElement> // Updated type to match <input>
  ) => {
    setSurgery(event.target.value);
  };

  // Huggingface pretrained model name and inference endpoint
  const SELECTED_SOUND_MODEL = {
    name: "Speechbrain - Ljspeech",
    url: "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits",
  };

  /**
   * Handles starting and stopping the speech recognition.
   */
  const handleOnRecord = () => {
    if (isActive) {
      recognitionRef.current?.stop();
      setIsActive(false);
      return;
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setErrorMessage(
        "Speech Recognition API is not supported in this browser."
      );
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    // recognitionRef.current.lang = 'en-US'; // Set language as needed
    // recognitionRef.current.interimResults = false;
    // recognitionRef.current.maxAlternatives = 1;

    recognitionRef.current.onstart = () => {
      setIsActive(true);
      setErrorMessage("");
    };

    recognitionRef.current.onend = () => {
      setIsActive(false);
    };

    recognitionRef.current.onresult = async (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript;
      setText(transcript);
      setIsLoading(true);
      setErrorMessage("");

      try {
        const response = await duringSurgery(transcript, patientHistory);
        setAIResponse(response.data);
      } catch (error) {
        console.error("Error fetching AI response:", error);
        setAIResponse("Sorry, there was an error processing your request.");
      } finally {
        setIsLoading(false);
      }
    };

    recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error("Speech Recognition Error:", event.error);
      setIsActive(false);
      setErrorMessage("Speech recognition error occurred. Please try again.");
    };

    recognitionRef.current.start();
  };

  /**
   * Generates audio from the AI response using the selected sound model.
   */
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const generateAndPlayAudio = async (text: string) => {
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: text,
          modelUrl: SELECTED_SOUND_MODEL.url,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate audio.");
      }

      const data = await response.arrayBuffer();
      const blob = new Blob([data], { type: "audio/mpeg" });
      const generatedAudioUrl = URL.createObjectURL(blob);
      setAudioUrl(generatedAudioUrl);
    } catch (error: any) {
      console.error("Error generating audio:", error);
      setErrorMessage(`Error generating audio: ${error.message}`);
    }
  };

  const handleFileUpload = async () => {
    if (!files.prescription || !files.scan || !files.labReport) {
      setUploadStatus("All three files are required.");
      return;
    }

    try {
      setUploadStatus("Uploading...");
      const response = await uploadPreSurgeryFiles(
        files,
        surgery,
        patientHistory
      );

      setAIResponse(response.data);
    } catch (error) {
      console.error("Error during file upload:", error);
      setUploadStatus("An error occurred during the upload.");
    }
  };

  /**
   * useEffect to watch for changes in aiResponse and trigger audio generation.
   */
  useEffect(() => {
    if (aiResponse.trim() !== "") {
      generateAndPlayAudio(aiResponse);
    }
    // Cleanup the audio URL when component unmounts or before setting a new one
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [aiResponse]);

  /**
   * useEffect to auto-play the audio when audioUrl changes.
   */
  useEffect(() => {
    if (audioUrl) {
      const audioElement = document.getElementById(
        "aiAudio"
      ) as HTMLAudioElement;
      if (audioElement) {
        audioElement.play().catch((error) => {
          console.error("Error playing audio:", error);
          setErrorMessage("Error playing audio. Please try again.");
        });
      }
    }
  }, [audioUrl]);

  return (
    <section className="max-w-md w-full flex flex-col gap-y-5 rounded-xl overflow-hidden mx-auto">
      <div className="bg-zinc-200 p-4">
        <div className="bg-blue-200 rounded-lg p-2 border-2 border-blue-300">
          <ul className="font-mono text-center font-bold text-blue-900 uppercase px-4 py-2 border border-blue-800 rounded">
            <li>&gt; SurgiAI</li>
          </ul>
        </div>
      </div>

      <RoutingButtons
        surgeryProcedure={surgeryProcedure}
        setSurgeryProcedure={setSurgeryProcedure}
      />

      <PatientHistory setPatientHistory={setPatientHistory} />

      {isActive && (
        <div className="w-full h-12 rounded-lg mb-8 flex items-center justify-center space-x-2">
          {/* Simulated waveform animation */}
          <div className="w-2 h-6 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-8 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-4 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-6 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-8 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-6 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-8 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-4 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-6 bg-blue-500 animate-pulse rounded-sm"></div>
          <div className="w-2 h-8 bg-blue-500 animate-pulse rounded-sm"></div>
        </div>
      )}

      {patientHistory.length > 0 && surgeryProcedure == false && (
        <>
          <button
            className={`w-full h-full flex items-center gap-x-2 justify-center uppercase font-semibold text-sm ${
              isActive ? "text-white bg-red-500" : "text-zinc-200 bg-zinc-900"
            } color-white py-3 rounded-sm`}
            onClick={handleOnRecord}
          >
            {isActive ? (
              <FaStop className="animate-pulse w-4 h-4" />
            ) : (
              <FaMicrophone className="w-4 h-4" />
            )}

            {isActive ? "Stop" : "Ask AI"}
          </button>

          <p>
            <strong>Surgeon Query:</strong> {text}
          </p>
          {isLoading ? (
            <p className="mb-4 animate-pulse text-sm text-gray-500">
              AI Response: Loading...
            </p>
          ) : (
            <p className="mb-4">
              <strong>AI Response:</strong> {aiResponse}
            </p>
          )}
        </>
      )}

      {surgeryProcedure == true && (
        <div className="flex flex-col">
          <input
            type="text"
            className="text-sm px-4 py-2 mb-2"
            placeholder="Enter operation you want to perform"
            onChange={handleOperationChange}
          />
          <FileUpload
            onFilesChange={(updatedFiles) => setFiles(updatedFiles)}
          />

          <button
            onClick={handleFileUpload}
            className="bg-blue-400 text-white px-4 py-2 rounded text-sm"
          >
            Upload Files and Generate Report
          </button>
        </div>
      )}

      {surgeryProcedure == true && (
        <p className="mb-4">
          <strong>AI Response:</strong> {aiResponse}
        </p>
      )}

      {/* Display Error Messages */}
      {errorMessage && (
        <div className="bg-red-100 text-red-700 p-2 rounded mt-2">
          {errorMessage}
        </div>
      )}

      {/* Audio Player */}
      {audioUrl && (
        <audio key={audioUrl} id="aiAudio" controls className="mt-4" autoPlay>
          <source type="audio/mpeg" src={audioUrl} />
          Your browser does not support the audio element.
        </audio>
      )}
    </section>
  );
};

export default VoiceInput;
