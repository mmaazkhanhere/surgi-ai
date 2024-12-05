import React, { useRef } from "react";

type Props = {
  setPatientHistory: (patientHistory: string) => void;
};

const PatientHistory = ({ setPatientHistory }: Props) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const handleHistoryChange = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setPatientHistory(event.target.value);
    autoResizeTextArea();
  };

  const autoResizeTextArea = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"; // Reset height
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`; // Set new height
    }
  };

  return (
    <section className="flex flex-col gap-y-2 w-full h-full">
      <label className="text-sm">Patient History</label>
      <textarea
        ref={textareaRef}
        className="max-h-60 h-full rounded-md text-sm p-2"
        placeholder="Enter the patient history"
        onChange={handleHistoryChange}
        style={{ overflow: "hidden" }} // Prevent scrollbar from appearing
      />
    </section>
  );
};

export default PatientHistory;
