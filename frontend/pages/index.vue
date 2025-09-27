<template>
  <div class="container mx-auto px-4 py-8">
    <h1 class="text-4xl font-bold mb-8 text-center text-pink-700">🎗️ Breast Cancer Mammography Detection</h1>
    <p class="text-center text-gray-600 mb-8">AI-powered breast cancer analysis from mammography images</p>

    <!-- System Information -->
    <div class="bg-gradient-to-r from-pink-50 to-rose-50 border-l-4 border-pink-500 shadow-lg rounded-lg p-6 mb-8">
      <h2 class="text-xl font-semibold mb-4 text-pink-800">🏥 Specialized Mammography Analysis</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div>
          <p><strong>Analysis:</strong> Suspicious vs Benign mammographic findings</p>
          <p><strong>Model:</strong> ResNet18 specialized for mammography</p>
        </div>
        <div>
          <p><strong>Training:</strong> Automatic training with mammography datasets</p>
          <p><strong>Focus:</strong> Radiological pattern recognition</p>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Upload Section -->
      <div class="bg-white shadow-lg rounded-lg p-6">
        <h2 class="text-xl font-semibold mb-4 text-pink-700">📤 Upload Mammography Image</h2>
        <div
          class="border-2 border-dashed border-pink-300 rounded-lg p-8 text-center hover:border-pink-400 transition-colors">
          <input type="file" @change="handleFileUpload" accept="image/*,.dcm" class="mb-4">
          <div class="text-gray-600">
            <svg class="mx-auto h-12 w-12 text-pink-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <p class="font-medium">Select a mammography image</p>
            <p class="text-sm text-gray-500 mt-2">Supported formats: JPG, PNG, TIFF, DICOM (.dcm)</p>
            <p class="text-sm text-pink-600 mt-1">� Optimized for mammographic screening images</p>
          </div>
        </div>

        <button @click="analyzeMammogram" :disabled="!selectedFile || isLoading"
          class="w-full mt-4 bg-pink-500 hover:bg-pink-700 disabled:bg-gray-300 text-white font-bold py-3 px-4 rounded-lg transition-colors">
          <span v-if="isLoading">🏥 Analyzing Mammogram...</span>
          <span v-else>🎗️ Analyze for Breast Cancer</span>
        </button>

        <!-- Selected File Info -->
        <div v-if="selectedFile" class="mt-4 p-3 bg-pink-50 rounded-lg border border-pink-200">
          <p class="text-sm font-medium text-pink-800">Selected File:</p>
          <p class="text-sm text-pink-700">{{ selectedFile.name }}</p>
          <p class="text-sm text-pink-600">Size: {{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB</p>
        </div>
      </div>

      <!-- Results Section -->
      <div class="bg-white shadow-lg rounded-lg p-6">
        <h2 class="text-xl font-semibold mb-4 text-pink-700">🔬 Breast Cancer Analysis Results</h2>

        <div v-if="prediction" class="space-y-4">
          <!-- Main Result -->
          <div :class="[
            'p-4 rounded-lg border-l-4',
            prediction.prediction === 'malignant' ? 'bg-red-50 border-red-400' : 'bg-green-50 border-green-400'
          ]">
            <div class="flex justify-between items-start">
              <div>
                <h3 class="font-semibold text-lg capitalize">
                  {{ prediction.prediction === 'malignant' ? '⚠️ Malignant' : '✅ Benign' }} Breast Tissue
                </h3>
                <p class="text-sm text-gray-600 mt-1">Confidence: {{ prediction.confidence?.toFixed(1) }}%</p>
              </div>
              <div :class="[
                'px-3 py-1 rounded-full text-sm font-medium',
                prediction.risk_level?.includes('HIGH') ? 'bg-red-100 text-red-800' :
                  prediction.risk_level?.includes('MODERATE') ? 'bg-yellow-100 text-yellow-800' :
                    prediction.risk_level?.includes('LOW') ? 'bg-green-100 text-green-800' :
                      'bg-gray-100 text-gray-800'
              ]">
                Risk: {{ prediction.risk_level?.replace(/_/g, ' ') || 'Unknown' }}
              </div>
            </div>
          </div>

          <!-- Breast Cancer Specific Interpretation -->
          <div class="bg-pink-50 p-4 rounded-lg border border-pink-200">
            <h4 class="font-medium text-pink-900 mb-2">🎗️ Breast Cancer Assessment:</h4>
            <p class="text-pink-800 text-sm">{{ prediction.interpretation }}</p>
          </div>

          <!-- Probability Breakdown -->
          <div class="bg-gray-50 p-4 rounded-lg">
            <h4 class="font-medium mb-3">🎯 Probability Breakdown:</h4>
            <div class="space-y-2">
              <div v-for="(prob, className) in prediction.probabilities" :key="className" class="flex items-center">
                <span class="w-20 text-sm capitalize">{{ className }}:</span>
                <div class="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                  <div :class="className === 'malignant' ? 'bg-red-500' : 'bg-green-500'"
                    class="h-2 rounded-full transition-all duration-500" :style="`width: ${prob}%`">
                  </div>
                </div>
                <span class="text-sm font-medium">{{ prob?.toFixed(1) }}%</span>
              </div>
            </div>
          </div>

          <!-- Breast Cancer Medical Disclaimer -->
          <div class="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
            <p class="text-yellow-800 text-sm font-medium">⚠️ Medical Disclaimer:</p>
            <p class="text-yellow-700 text-sm mt-1">
              This AI analysis is specialized for breast cancer research and educational purposes only.
              Always consult with qualified oncologists and pathologists for medical diagnosis and treatment decisions.
            </p>
          </div>
        </div>

        <!-- No Results -->
        <div v-else class="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
          <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p class="text-gray-600">Upload and analyze a breast tissue image to see results</p>
        </div>
      </div>
    </div>

    <!-- Breast Cancer Model Information Section -->
    <div class="mt-8 bg-white shadow-lg rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4 text-pink-700">🏥 Breast Cancer Detection System Information</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <button @click="getSystemInfo"
            class="bg-pink-500 hover:bg-pink-700 text-white font-bold py-2 px-4 rounded mb-4">
            📊 View System Info
          </button>
          <button @click="checkModelStatus"
            class="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded mb-4 ml-2">
            🔍 Check Model Status
          </button>
        </div>
      </div>

      <div v-if="modelStatus" class="mt-4 p-4 bg-pink-50 border border-pink-200 rounded-lg">
        <h3 class="font-semibold text-pink-800">🎗️ Breast Cancer Model Status</h3>
        <div class="text-sm text-pink-700 mt-2 space-y-1">
          <p><strong>Model Type:</strong> {{ modelStatus.model_type }}</p>
          <p><strong>Cancer Type:</strong> {{ modelStatus.cancer_type || 'Breast Cancer' }}</p>
          <p><strong>Status:</strong> {{ modelStatus.model_trained ? '✅ Ready' : '❌ Not Trained' }}</p>
          <p><strong>Device:</strong> {{ modelStatus.device }}</p>
          <p><strong>Classes:</strong> {{ modelStatus.classes?.join(', ') }}</p>
          <p v-if="modelStatus.training_accuracy"><strong>Training Accuracy:</strong> {{
            modelStatus.training_accuracy?.toFixed(1) }}%</p>
          <p v-if="modelStatus.validation_accuracy"><strong>Validation Accuracy:</strong> {{
            modelStatus.validation_accuracy?.toFixed(1) }}%</p>
        </div>
      </div>

      <div v-if="systemInfo" class="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-lg">
        <h3 class="font-semibold text-purple-800">ℹ️ System Information</h3>
        <pre class="text-sm text-purple-700 mt-2 whitespace-pre-wrap">{{ systemInfo }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
const selectedFile = ref(null)
const prediction = ref(null)
const isLoading = ref(false)
const modelStatus = ref(null)
const systemInfo = ref(null)

// Backend API URL
const API_BASE_URL = 'http://localhost:8000'

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
  prediction.value = null
}

const analyzeMammogram = async () => {
  if (!selectedFile.value) return

  isLoading.value = true
  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    if (result.status === 'success') {
      prediction.value = result
      console.log('Mammography analysis result:', result)
    } else {
      throw new Error('Mammography analysis failed')
    }
  } catch (error) {
    console.error('Error analyzing mammography:', error)
    alert('Error analyzing mammography image. Please try again.')
  } finally {
    isLoading.value = false
  }
}

const getSystemInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/model/info`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const result = await response.json()
    systemInfo.value = result.system_info
  } catch (error) {
    console.error('Error fetching system information:', error)
    alert('Error fetching system information')
  }
}

const checkModelStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/model/status`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const result = await response.json()
    modelStatus.value = result
  } catch (error) {
    console.error('Error checking model status:', error)
    alert('Error checking model status')
  }
}

// Check initial model status
onMounted(() => {
  checkModelStatus()
})
</script>