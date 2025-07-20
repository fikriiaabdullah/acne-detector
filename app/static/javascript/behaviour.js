// Script webcam & capture untuk Deteksi Jerawat Otomatis

// Ambil elemen-elemen yang diperlukan
const video = document.getElementById("webcam")
const canvas = document.getElementById("canvas")
const captureBtn = document.getElementById("capture-btn")
const previewImg = document.getElementById("preview")
const capturedInput = document.getElementById("captured-image")
const uploadForm = document.getElementById("upload-form")
let stream = null

// Tambah tombol ulangi foto
defineRetakeButton()
let retakeBtn = document.getElementById("retake-btn")

function defineRetakeButton() {
  if (!document.getElementById("retake-btn")) {
    const btn = document.createElement("button")
    btn.type = "button"
    btn.className = "btn btn-secondary btn-main"
    btn.id = "retake-btn"
    btn.textContent = "Ulangi Foto"
    btn.style.display = "none"
    captureBtn.parentNode.insertBefore(btn, canvas.nextSibling)
  }
}

retakeBtn = document.getElementById("retake-btn")

// Mulai webcam dengan constraints yang lebih baik
function startWebcam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Request higher quality video for better face detection
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user", // Front camera
      },
    }

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((s) => {
        stream = s
        video.srcObject = stream
        video.play()

        // Add face detection overlay
        addFaceDetectionOverlay()
      })
      .catch((err) => {
        alert("Tidak dapat mengakses kamera: " + err.message)
      })
  } else {
    alert("Browser Anda tidak mendukung akses kamera.")
  }
}

// Add face detection overlay to guide user
function addFaceDetectionOverlay() {
  const overlay = document.createElement("div")
  overlay.id = "face-overlay"
  overlay.style.cssText = `
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 200px;
    height: 250px;
    border: 3px solid #39b0e0;
    border-radius: 50%;
    pointer-events: none;
    z-index: 10;
  `

  const overlayText = document.createElement("div")
  overlayText.textContent = "Posisikan wajah di dalam oval"
  overlayText.style.cssText = `
    position: absolute;
    top: -40px;
    left: 50%;
    transform: translateX(-50%);
    color: #39b0e0;
    font-weight: bold;
    font-size: 14px;
    text-align: center;
    white-space: nowrap;
  `

  overlay.appendChild(overlayText)
  video.parentNode.style.position = "relative"
  video.parentNode.appendChild(overlay)
}

startWebcam()

// Fungsi untuk menangkap gambar dari video dengan kualitas lebih baik
captureBtn.addEventListener("click", (e) => {
  e.preventDefault()

  // Set canvas size to match video dimensions
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  const ctx = canvas.getContext("2d")
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

  // Get high quality image data
  const dataUrl = canvas.toDataURL("image/jpeg", 0.9)
  previewImg.src = dataUrl
  previewImg.style.display = "block"
  capturedInput.value = dataUrl

  // Hide video and show preview
  video.style.display = "none"
  document.getElementById("face-overlay")?.remove()
  captureBtn.style.display = "none"
  retakeBtn.style.display = "inline-block"

  // Show success message
  showMessage("Foto berhasil diambil! Silakan isi data dan klik Analisis Jerawat.", "success")
})

// Fungsi ulangi foto
retakeBtn.addEventListener("click", (e) => {
  e.preventDefault()
  previewImg.style.display = "none"
  capturedInput.value = ""
  video.style.display = "block"
  captureBtn.style.display = "inline-block"
  retakeBtn.style.display = "none"

  // Re-add face detection overlay
  addFaceDetectionOverlay()
})

// Validasi sebelum submit
uploadForm.addEventListener("submit", (e) => {
  if (!capturedInput.value) {
    alert("Silakan ambil foto terlebih dahulu!")
    e.preventDefault()
    return false
  }

  // Show loading message
  showMessage("Sedang menganalisis foto... Mohon tunggu.", "info")

  // Disable submit button to prevent double submission
  const submitBtn = uploadForm.querySelector('button[type="submit"]')
  submitBtn.disabled = true
  submitBtn.textContent = "Menganalisis..."
})

// Function to show messages to user
function showMessage(message, type) {
  // Remove existing message
  const existingMessage = document.getElementById("user-message")
  if (existingMessage) {
    existingMessage.remove()
  }

  const messageDiv = document.createElement("div")
  messageDiv.id = "user-message"
  messageDiv.className = `alert alert-${type === "success" ? "success" : type === "error" ? "danger" : "info"}`
  messageDiv.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    max-width: 300px;
    padding: 10px 15px;
    border-radius: 5px;
    font-weight: 500;
  `
  messageDiv.textContent = message

  document.body.appendChild(messageDiv)

  // Auto remove after 5 seconds
  setTimeout(() => {
    messageDiv.remove()
  }, 5000)
}

function uploadStarted() {
  if (window.androidInterface) {
    window.androidInterface.showToast("Uploading image...")
  }
}

document.getElementById("currentYear").innerHTML = new Date().getFullYear()
