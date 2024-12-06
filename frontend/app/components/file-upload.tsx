"use client";
import React, { useState } from "react";
import { preSurgeryMedicine } from "../actions/medicine-upload";

const FileUpload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      setUploadStatus("No file selected.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      //   const response = await fetch("http://127.0.0.1:8000/upload/", {
      //     method: "POST",
      //     body: formData,
      //   });

      const response = await preSurgeryMedicine(formData);

      if (response.status == 200) {
        setUploadStatus("File uploaded successfully!");
      } else {
        setUploadStatus("File upload failed.");
      }
    } catch (error) {
      setUploadStatus("An error occurred during the upload.");
      console.error(error);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <input
        type="file"
        accept="application/pdf"
        onChange={handleFileChange}
        className="mb-4"
      />
      <button
        onClick={handleFileUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Upload File
      </button>
      {uploadStatus && <p className="mt-4">{uploadStatus}</p>}
    </div>
  );
};

export default FileUpload;
