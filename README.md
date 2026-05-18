# 🏥 Edge AI-Based Smart Patient Monitoring and Assistance System



# 🧾 Introduction
The **Edge AI-Based Smart Patient Monitoring and Assistance System** is an intelligent healthcare support solution designed to help elderly individuals, post-operative patients, and people with limited mobility communicate their needs effectively. The system combines **Edge AI**, **TinyML**, voice recognition, and manual input methods to provide fast, reliable, and low-latency assistance without depending on cloud connectivity.

---

# 🧭 Overview
This project uses an **SIWG917 Edge AI MCU** integrated with voice and button-based inputs to identify patient requests such as:
- Food
- Water
- Restroom Assistance
- Emergency
- Pain

Using an on-device TinyML keyword classification model, the system processes patient commands locally and provides instant caregiver alerts through audio output.

The system is designed to be:
- Affordable
- Reliable
- Scalable
- Easy to use in hospitals and home-care environments

---

# 🚀 Key Features
- 🎙️ Voice-based patient request recognition
- 🔘 Push-button manual assistance input
- 🧠 TinyML-powered keyword classification
- ⚡ Real-time on-device inference
- 🔊 Audio alerts through speaker output
- ☁️ No cloud dependency
- 📉 Low latency response system
- 💾 Memory-efficient embedded AI implementation
- 🔋 Edge AI processing for reliable operation

---

## Machine Learning Details

| Parameter | Value |
|---|---|
| Model Type | TinyML Keyword Classification |
| Input Shape | 1 × 98 × 1 × 104 |
| Output Classes | Food, Water, Restroom, Emergency, Pain, Noise, Unknown, Silence |
| Input Data Type | int8 |
| Flash Size | 175.2 KB |
| Processing Type | On-Device Edge AI Inference |

---

# 📡 Sensor Details

| Component | Purpose |
|---|---|
| Microphone | Voice input capture |
| Push Button | Manual patient request input |
| SIWG917 MCU | Edge AI processing |
| DF Mini MP3 Player | Audio playback |
| Speaker | Caregiver alert output |

---

# 📊 Block Diagram

<img width="6029" height="1920" alt="Picture1" src="https://github.com/user-attachments/assets/93d86d6d-98d8-4411-b9d1-04907038f2fa" />

---

# 🔌 Pin Connections

| Component | Connected To |
|---|---|
| Microphone Output | MCU Audio Input Pin |
| Push Button | GPIO Input Pin |
| DF Mini MP3 TX/RX | MCU UART Pins |
| Speaker | DF Mini MP3 Output |
| Power Supply | MCU and Modules |

---

# 📽️ Application Videos

https://github.com/user-attachments/assets/e544ae3d-50d0-41f0-9165-4ab8ca60558e

---

# 👥 Contributors

- Mohan Raj S — mohanraj.s2023ece@sece.ac.in  
- Nashrin Fathima K — nashrinfathima.k2023ece@sece.ac.in

### Team Name
**Creative Crew**
