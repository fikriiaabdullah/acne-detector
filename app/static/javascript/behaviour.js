// Script webcam & capture untuk Deteksi Jerawat Otomatis

// Ambil elemen-elemen yang diperlukan
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const previewImg = document.getElementById('preview');
const capturedInput = document.getElementById('captured-image');
const uploadForm = document.getElementById('upload-form');
let stream = null;

// Tambah tombol ulangi foto
defineRetakeButton();
let retakeBtn = document.getElementById('retake-btn');

function defineRetakeButton() {
  if (!document.getElementById('retake-btn')) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-secondary btn-main';
    btn.id = 'retake-btn';
    btn.textContent = 'Ulangi Foto';
    btn.style.display = 'none';
    captureBtn.parentNode.insertBefore(btn, canvas.nextSibling);
  }
}

retakeBtn = document.getElementById('retake-btn');

// Mulai webcam
function startWebcam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function(s) {
        stream = s;
        video.srcObject = stream;
        video.play();
      })
      .catch(function(err) {
        alert('Tidak dapat mengakses kamera: ' + err.message);
      });
  } else {
    alert('Browser Anda tidak mendukung akses kamera.');
  }
}
startWebcam();

// Fungsi untuk menangkap gambar dari video
captureBtn.addEventListener('click', function(e) {
  e.preventDefault();
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg');
  previewImg.src = dataUrl;
  previewImg.style.display = 'block';
  capturedInput.value = dataUrl;
  video.style.display = 'none';
  captureBtn.style.display = 'none';
  retakeBtn.style.display = 'inline-block';
});

// Fungsi ulangi foto
retakeBtn.addEventListener('click', function(e) {
  e.preventDefault();
  previewImg.style.display = 'none';
  capturedInput.value = '';
  video.style.display = 'block';
  captureBtn.style.display = 'inline-block';
  retakeBtn.style.display = 'none';
});

// Validasi sebelum submit: pastikan sudah ambil foto
uploadForm.addEventListener('submit', function(e) {
  if (!capturedInput.value) {
    alert('Silakan ambil foto terlebih dahulu!');
    e.preventDefault();
    return false;
  }
});

function uploadStarted() {
  window.androidInterface.showToast("Uploading image...")
}

document.getElementById("currentYear").innerHTML = new Date().getFullYear();

