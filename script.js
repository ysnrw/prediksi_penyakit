let model;
let dataset = [];

const fallbackDataset = [
  {
    "keluhan": "demam batuk pilek sakit tenggorokan",
    "penyakit": "Flu",
    "solusi": "Istirahat cukup, minum air hangat, konsumsi vitamin C"
  },
  {
    "keluhan": "nyeri ulu hati mual perut kembung",
    "penyakit": "Maag",
    "solusi": "Makan porsi kecil tapi sering, hindari makanan pedas/berminyak"
  }
];

const showLoading = () => {
  Swal.fire({
    title: 'Memproses',
    html: 'Training model... <b>0%</b>',
    allowOutsideClick: false,
    didOpen: () => Swal.showLoading()
  });
};

const hideLoading = () => Swal.close();

const showAlert = (title, text, icon = 'info') => {
  return Swal.fire({
    title,
    text,
    icon,
    confirmButtonText: 'Mengerti',
    confirmButtonColor: '#3498db'
  });
};


const encodeText = (text, vocabulary) => {
  const words = text.toLowerCase().split(/\s+/);
  const encoded = new Array(vocabulary.length).fill(0);
  
  words.forEach(word => {
    const index = vocabulary.indexOf(word);
    if (index !== -1) encoded[index] = 1;
  });
  
  return encoded;
};

const initModel = async () => {
  try {
    await showLoading();
    
    model = tf.sequential();
    
    const vocabulary = [...new Set(dataset.flatMap(item => 
      item.keluhan.toLowerCase().split(/\s+/)
    ))];
    
    model.vocabulary = vocabulary;
    
    model.add(tf.layers.dense({
      units: 32,
      activation: 'relu',
      inputShape: [vocabulary.length]
    }));
    
    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu'
    }));
    
    model.add(tf.layers.dense({
      units: dataset.length,
      activation: 'softmax'
    }));
    
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    const xs = tf.tensor2d(
      dataset.map(item => encodeText(item.keluhan, vocabulary))
    );
    
    const ys = tf.tensor2d(
      dataset.map((_, i) => {
        const arr = new Array(dataset.length).fill(0);
        arr[i] = 1;
        return arr;
      })
    );
    
    await model.fit(xs, ys, {
      epochs: 200,
      batchSize: 8,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress = ((epoch + 1) / 200 * 100).toFixed(2);
          const progressText = `Training model... <b>${progress}%</b><br>Loss: ${logs.loss.toFixed(4)}`;

          const htmlContainer = Swal.getHtmlContainer();
          if (htmlContainer) {
            htmlContainer.innerHTML = progressText;
          }
        }
      }
    });
    
    xs.dispose();
    ys.dispose();
    
    await hideLoading();
    console.log('Model siap digunakan');
    
  } catch (error) {
    await hideLoading();
    await showAlert('Error', `Gagal inisialisasi model: ${error.message}`, 'error');
    throw error;
  }
};

const loadDataset = async () => {
  try {
    await showLoading();
    
    const response = await fetch('dataset.json');
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    
    dataset = await response.json();
    console.log('Dataset loaded:', dataset.length, 'entries');
    
    await hideLoading();
  } catch (error) {
    dataset = fallbackDataset;
    await hideLoading();
    await showAlert(
      'Peringatan', 
      'Menggunakan dataset bawaan', 
      'warning'
    );
    console.warn('Using fallback dataset');
  }
};

const predictDisease = async () => {
  const inputText = document.getElementById('keluhan-input').value.trim();
  
  if (!inputText) {
    await showAlert('Peringatan', 'Masukkan keluhan terlebih dahulu', 'warning');
    return;
  }
  
  if (!model) {
    await showAlert('Error', 'Model belum siap. Silahkan refresh halaman', 'error');
    return;
  }
  
  try {
    await showLoading();
    
    const encodedInput = encodeText(inputText, model.vocabulary);
    const inputTensor = tf.tensor2d([encodedInput]);
    
    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.data();
    
    inputTensor.dispose();
    prediction.dispose();

    const maxIndex = predictionData.indexOf(Math.max(...predictionData));
    const confidence = (predictionData[maxIndex] * 100).toFixed(2);
    const result = dataset[maxIndex];

    document.getElementById('predicted-disease').textContent = result.penyakit;
    document.getElementById('confidence').textContent = `Tingkat Kepercayaan: ${confidence}%`;
    document.getElementById('treatment-text').textContent = result.solusi;
    document.querySelector('.result-section').classList.remove('hidden');
    
    await hideLoading();
    await showAlert(
      'Hasil Diagnosa',
      `Teridentifikasi: ${result.penyakit} (${confidence}% keyakinan)`,
      'success'
    );
    
  } catch (error) {
    await hideLoading();
    await showAlert(
      'Error', 
      `Gagal memproses: ${error.message}`, 
      'error'
    );
    console.error('Prediction error:', error);
  }
};

document.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadDataset();
    await initModel();
    
    document.getElementById('predict-btn').addEventListener('click', predictDisease);
    
    await showAlert(
      'Aplikasi Siap', 
      'Sistem diagnosa sudah siap digunakan. Masukkan keluhan Anda.',
      'info'
    );
    
  } catch (error) {
    await showAlert(
      'Error Kritis', 
      'Aplikasi tidak dapat dimulai. Silahkan refresh halaman.',
      'error'
    );
    console.error('Initialization error:', error);
  }
});