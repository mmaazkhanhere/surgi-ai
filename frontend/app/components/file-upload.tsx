"use client";
import React, { useState } from "react";
import { preSurgeryMedicine } from "@/app/actions/medicine-upload";
import { preSurgeryLabReports } from "@/app/actions/lab-report-upload";
import { preSurgeryScans } from "@/app/actions/scan-upload";

const FileUpload = () => {
  const [files, setFiles] = useState({
    prescription: null as File | null,
    scan: null as File | null,
    labReport: null as File | null,
  });
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
    type: string
  ) => {
    if (event.target.files) {
      setFiles((prevFiles) => ({
        ...prevFiles,
        [type]: event.target.files![0],
      }));
    }
  };

  const handleFileUpload = async () => {
    if (!files.prescription || !files.scan || !files.labReport) {
      setUploadStatus("All three images are required.");
      return;
    }

    try {
      setUploadStatus("Uploading...");

      // Upload prescription
      const prescriptionResponse = await preSurgeryMedicine(files.prescription);
      if (prescriptionResponse.status !== 200) {
        throw new Error("Prescription upload failed.");
      }

      // Upload scan
      const scanResponse = await preSurgeryScans(files.scan);
      if (scanResponse.status !== 200) {
        throw new Error("Scan upload failed.");
      }

      // Upload lab report
      const labReportResponse = await preSurgeryLabReports(files.labReport);
      if (labReportResponse.status !== 200) {
        throw new Error("Lab report upload failed.");
      }

      setUploadStatus("All files uploaded successfully!");
    } catch (error) {
      setUploadStatus("An error occurred during the upload.");
      console.error(error);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div className="mb-4">
        <label className="block mb-2 text-sm">Upload Prescription</label>
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg, image/webp"
          onChange={(e) => handleFileChange(e, "prescription")}
          className="text-xs"
        />
      </div>

      <div className="mb-4">
        <label className="block mb-2 text-sm">Upload Scan</label>
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg, image/webp"
          onChange={(e) => handleFileChange(e, "scan")}
          className="text-xs"
        />
      </div>

      <div className="mb-4">
        <label className="block mb-2 text-sm">Upload Lab Report</label>
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg, image/webp"
          onChange={(e) => handleFileChange(e, "labReport")}
          className="text-xs"
        />
      </div>

      <button
        onClick={handleFileUpload}
        className="bg-blue-400 text-white px-4 py-2 rounded text-sm"
      >
        Upload Files
      </button>
      {uploadStatus && <p className="mt-4">{uploadStatus}</p>}
    </div>
  );
};

export default FileUpload;
