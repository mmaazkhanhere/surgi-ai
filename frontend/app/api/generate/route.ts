import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest): Promise<Response> {
  try {
    const requestBody = await request.json();

    if (!requestBody.modelUrl) {
      return new NextResponse("Missing 'modelUrl' field in the request body.", {
        status: 400,
      });
    }

    if (!requestBody.input) {
      return new NextResponse("Missing 'input' field in the request body.", {
        status: 400,
      });
    }

    if (!process.env.HUGGING_FACE_TOKEN) {
      return new NextResponse(
        "Missing 'HUGGING_FACE_TOKEN' field in the request body.",
        { status: 400 }
      );
    }

    const { modelUrl, input } = requestBody;

    // Make a POST request to the Hugging Face Inference API
    const hfResponse = await fetch(modelUrl, {
      headers: {
        Authorization: `Bearer ${process.env.HUGGING_FACE_TOKEN}`,
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify({ inputs: input }),
    });

    // Check if the Hugging Face API responded successfully
    if (!hfResponse.ok) {
      const errorDetails = await hfResponse.text(); // or hfResponse.json() if it's JSON
      console.error("Hugging Face API Error:", hfResponse.status, errorDetails);
      return new Response(
        JSON.stringify({
          error: "Hugging Face API request failed.",
          details: errorDetails,
        }),
        {
          status: hfResponse.status,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // Get the generated audio data as an ArrayBuffer
    const audioData = await hfResponse.arrayBuffer();

    // Return the audio data with appropriate headers
    return new Response(audioData, {
      headers: {
        "Content-Type": "audio/mpeg", // Adjust if necessary
      },
    });
  } catch (error: any) {
    console.error("Server Error in /api/generate:", error);
    return new Response(
      JSON.stringify({
        error: "Internal Server Error.",
        details: error.message,
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
