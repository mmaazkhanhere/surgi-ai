"use client";
import React, { useState } from "react";

type FileUploadProps = {
  onFilesChange: (files: Record<string, File | null>) => void;
};

const FileUpload = ({ onFilesChange }: FileUploadProps) => {
  const [files, setFiles] = useState<Record<string, File | null>>({
    prescription: null,
    scan: null,
    labReport: null,
  });

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
    type: string
  ) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setFiles((prevFiles) => {
        const updatedFiles = { ...prevFiles, [type]: file };
        onFilesChange(updatedFiles); // Notify the parent component
        return updatedFiles;
      });
    }
  };

  return (
    <div className="flex flex-col items-center gap-y-4">
      {/* Prescription Upload */}
      <div className="w-full">
        <label className="block mb-2 text-sm">Upload Prescription</label>
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg, image/webp"
          onChange={(e) => handleFileChange(e, "prescription")}
          className="text-xs"
        />
        {files.prescription && (
          <p className="text-sm mt-2">Selected: {files.prescription.name}</p>
        )}
      </div>

      {/* Scan Upload */}
      <div className="w-full">
        <label className="block mb-2 text-sm">Upload Scan</label>
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg, image/webp"
          onChange={(e) => handleFileChange(e, "scan")}
          className="text-xs"
        />
        {files.scan && (
          <p className="text-sm mt-2">Selected: {files.scan.name}</p>
        )}
      </div>

      {/* Lab Report Upload */}
      <div className="w-full">
        <label className="block mb-2 text-sm">Upload Lab Report</label>
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg, image/webp"
          onChange={(e) => handleFileChange(e, "labReport")}
          className="text-xs"
        />
        {files.labReport && (
          <p className="text-sm mt-2">Selected: {files.labReport.name}</p>
        )}
      </div>
    </div>
  );
};

export default FileUpload;
