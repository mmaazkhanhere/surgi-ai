# <div align="center">🩺 AI Surgical Assistant</div>

<div align="center">
  Revolutionizing surgery with real-time AI-powered guidance and automation.
</div>

---

## Table of Contents
1. [About the Project](#about-the-project)
2. [Tech Stack](#tech-stack)
3. [Features](#features)
4. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running Tests](#running-tests)
    - [Run Locally](#run-locally)
    - [Deployment](#deployment)
5. [Usage](#usage)
6. [Roadmap](#roadmap)
7. [Contributing](#contributing)
8. [Code of Conduct](#code-of-conduct)
9. [FAQ](#faq)
10. [License](#license)
11. [Contact](#contact)
12. [Acknowledgements](#acknowledgements)

---

## <a name="about-the-project"></a> 📝 About the Project

The **AI Surgical Assistant** is an advanced AI-driven system designed to support surgeons through every phase of surgery: pre-surgery, during surgery, and post-surgery. This tool provides real-time guidance, automates report generation, and aims to reduce human errors through voice-activated commands.

<div align="center">
  <!-- Insert screenshot here -->
  <img src="path-to-screenshot" alt="AI Surgical Assistant Screenshot" width="800"/>
</div>

---

## <a name="tech-stack"></a> 🛠️ Tech Stack

The AI Surgical Assistant utilizes the following technologies:

- **Programming Language**: Python
- **Frontend**: HTML, CSS
- **Backend Framework**: Flask
- **AI Model**: LangChain, LLaMA 2, CrewAI Agents
- **Vector Database**: Pinecone
- **Natural Language Processing**: Google Gemini Pro (LLM), Google’s embedding-001 model
- **Deployment**: IBM watsonx Granite Model

---

## <a name="features"></a> 🚀 Features

- **Pre-Surgery Report Generation**:
    - Automatically generates reports based on surgery name, prescriptions, test results, and scan reports.

- **Real-Time Voice-Activated Interaction**:
    - Hands-free voice commands for interaction during surgery.
    - Customizable wake-up phrases like "I have a question."

- **Surgery Guidance & Query Handling**:
    - Provides immediate answers to surgery-related queries without manual input.

- **Post-Surgery Report Generation**:
    - Creates detailed post-op reports based on patient condition, surgery details, and surgeon's observations.

- **Error Reduction**:
    - Aims to minimize human errors and enhance surgical outcomes with real-time insights.

---

## <a name="getting-started"></a> ⚙️ Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Surgical-Assistant.git
2. Navigate to the project directory:
   ```bash
   cd AI-Surgical-Assistant
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
### Running Tests
#### Run Locally
1. Start the application:
   ```bash
   python app.py
2. Open your browser and navigate to http://localhost:8501 to access the application.
  
### Deployment
To deploy the AI Surgical Assistant:

1. Build and deploy the application using a cloud service provider or server of your choice.

2. For example, if deploying on Heroku:
   ```bash
   heroku create
  git push heroku main

3. Set up environment variables and configuration as needed.
   
## <a name="usage"></a> 🛠️ Usage
1. Use voice commands to interact with the system during surgery, such as "I have a question" to activate and "That's it" to close the session.
2. Query the assistant for real-time guidance throughout the procedure.
3. Generate pre- and post-surgery reports automatically based on system feedback.

## <a name="roadmap"></a> 🚀 Roadmap
Q4 2024: Add support for additional languages.
Q1 2025: Integrate with advanced diagnostic tools.
Q2 2025: Expand the system to support multiple types of surgeries.

## <a name="contributing"></a> 🤝 Contributing
We welcome contributions to enhance the AI Surgical Assistant! Please:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/new-feature).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature/new-feature).
5. Open a pull request.

### Code of Conduct

We expect all contributors to follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a positive and respectful community environment.

## <a name="faq"></a> ❓ FAQ

**Q: How do I report issues?**

A: Please open an issue on the GitHub repository.

**Q: Can I use this project for commercial purposes?**

A: Review the License section for details on usage rights.
   
## <a name="license"></a> 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## <a name="contact"></a> 📫 Contact
For any questions or feedback, please contact:

**Email**: your-email@example.com<br>
**GitHub**: your-github-username

## <a name="acknowledgements"></a> 🙏 Acknowledgements
Thanks to all contributors and supporters of this project.
Special thanks to the developers and researchers whose work inspired this project.




