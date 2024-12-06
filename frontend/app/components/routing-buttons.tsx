import React from "react";

type Props = {
  surgeryProcedure: boolean;
  setSurgeryProcedure: (procedure: boolean) => void;
};

const RoutingButtons = ({ surgeryProcedure, setSurgeryProcedure }: Props) => {
  return (
    <div className="flex items-center justify-between">
      <button
        onClick={() => setSurgeryProcedure(false)}
        className={`${
          surgeryProcedure == false
            ? "bg-blue-200 border border-black"
            : "border border-black"
        } px-4 py-2 rounded-lg text-sm hover:bg-blue-200 hover:transition-all hover:duration-500`}
      >
        Surgery Assistant
      </button>

      <button
        onClick={() => setSurgeryProcedure(true)}
        className={`${
          surgeryProcedure == true
            ? "bg-blue-200 border border-black"
            : "border-black border"
        } px-4 py-2 rounded-lg text-sm hover:bg-blue-200 hover:transition-all hover:duration-500`}
      >
        Surgery Procedure Generation
      </button>
    </div>
  );
};

export default RoutingButtons;
