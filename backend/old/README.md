# <div align="center"><a href="https://surgi-ai.streamlit.app/">🩺 AI Surgical Assistant</a></div>

<div align="center">
  Revolutionizing surgery with real-time AI-powered guidance and automation.
</div>

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Tech Stack](#tech-stack)
4. [Installation](#installation)
5. [Usage](#usage)

---

> **Note:** Streamlit Cloud doesn't support real time voice listening. To experience during surgery voice chat, clone and run it locally

---
## <a name="overview"></a> 📝 Overview

The **AI Surgical Assistant** is an innovative AI-powered system designed to assist surgeons throughout the entire surgical process: **pre-surgery**, **during surgery**, and **post-surgery**. It provides real-time guidance, automates critical tasks like report generation, and reduces the risks of surgical errors through voice-activated commands.

Surgeons can interact with the system using customizable wake-up phrases and receive immediate, actionable feedback to ensure optimal patient outcomes.

### Why it matters:
- **Surgical errors** are one of the leading causes of complications in the operating room.
- Our system helps mitigate human performance deficiencies by providing critical, data-driven insights in real time.

---

## <a name="key-features"></a> 🚀 Key Features

- **Pre-Surgery Report Generation**:
    - Automatically generates reports based on the surgery name, prescriptions, test results, and scan reports.
    
- **Real-Time Voice-Activated Interaction**:
    - Hands-free voice command during surgery.
    - Customizable wake-up phrases like "I have a question" to trigger interaction.
    
- **Surgery Guidance & Answering Queries**:
    - Provides real-time answers to surgery-related questions without the need for manual interaction.

- **Post-Surgery Report Generation**:
    - Collects data on patient condition, surgery details, and surgeon's observations to create a comprehensive post-op report.

- **Error Reduction**:
    - Aims to reduce human errors and improve surgery outcomes by offering real-time feedback and guidance.

---

## <a name="tech-stack"></a> 🛠️ Tech Stack

The AI Surgical Assistant is built using cutting-edge technologies:

- **Programming Language**: Python
- **Frontend**: Streamlit (Python)
- **Backend Framework**: Python
- **AI Model**: LangChain, LLaMA 2, CrewAI Agents
- **Vector Database**: Pinecone
- **Report Generation**: AI-powered automation based on patient data and surgery details
- **Deployment**: Streamlit Cloud

---

## <a name="installation"></a> ⚙️ Installation

To get started with the AI Surgical Assistant:

1. Clone the repository:
   ```bash
   git clone https://github.com/mmaazkhanhere/surgi-ai.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Add environment variables. Contact us to share relevant env variables

4. Run the application:
   ```bash
   streamlit run main.py
   
   
## <a name="usage"></a> 🚀 Usage
1. Use the customizable voice commands like "I have a question" to activate the system during surgery.
2. Query the AI assistant for real-time answers and guidance throughout the surgical procedure.
3. Automatically generate both pre and post surgery reports based on real-time data and feedback.
