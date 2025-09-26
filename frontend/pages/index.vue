<template>
  <div class="container mx-auto px-4 py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">Image Classifier</h1>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
      <!-- Upload Section -->
      <div class="bg-white shadow-lg rounded-lg p-6">
        <h2 class="text-xl font-semibold mb-4">Upload Image for Prediction</h2>
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input type="file" @change="handleFileUpload" accept="image/*" class="mb-4">
          <p class="text-gray-600">Select an image to classify</p>
        </div>
        <button @click="predictImage" :disabled="!selectedFile"
          class="w-full mt-4 bg-blue-500 hover:bg-blue-700 disabled:bg-gray-300 text-white font-bold py-2 px-4 rounded">
          Predict
        </button>
      </div>

      <!-- Results Section -->
      <div class="bg-white shadow-lg rounded-lg p-6">
        <h2 class="text-xl font-semibold mb-4">Prediction Results</h2>
        <div v-if="prediction" class="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 class="font-semibold text-green-800">Prediction:</h3>
          <p class="text-green-700">{{ prediction.result }}</p>
          <p class="text-sm text-green-600 mt-2">Confidence: {{ prediction.confidence }}%</p>
        </div>
        <div v-else class="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <p class="text-gray-600">Upload an image to see prediction results</p>
        </div>
      </div>
    </div>

    <!-- Training Section -->
    <div class="mt-8 bg-white shadow-lg rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Model Training</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Training Dataset</label>
          <input type="file" @change="handleTrainingDataUpload" multiple accept="image/*" class="w-full">
        </div>
        <div>
          <button @click="trainModel" :disabled="!trainingFiles.length"
            class="bg-green-500 hover:bg-green-700 disabled:bg-gray-300 text-white font-bold py-2 px-4 rounded">
            Train Model
          </button>
        </div>
      </div>
      <div v-if="trainingStatus" class="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <p class="text-blue-800">{{ trainingStatus }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
const selectedFile = ref(null)
const prediction = ref(null)
const trainingFiles = ref([])
const trainingStatus = ref('')
const config = useRuntimeConfig()

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
  prediction.value = null
}

const handleTrainingDataUpload = (event) => {
  trainingFiles.value = Array.from(event.target.files)
}

const predictImage = async () => {
  if (!selectedFile.value) return

  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)

    // TODO: Implement API call to backend
    console.log('Predicting image:', selectedFile.value.name)

    // Placeholder response
    prediction.value = {
      result: 'Sample Classification',
      confidence: 85.5
    }
  } catch (error) {
    console.error('Error predicting image:', error)
  }
}

const trainModel = async () => {
  if (!trainingFiles.value.length) return

  try {
    const formData = new FormData()
    trainingFiles.value.forEach((file, index) => {
      formData.append(`training_image_${index}`, file)
    })

    trainingStatus.value = 'Starting model training...'

    // TODO: Implement API call to backend
    console.log('Training model with', trainingFiles.value.length, 'files')

    // Placeholder status update
    setTimeout(() => {
      trainingStatus.value = 'Model training completed successfully!'
    }, 2000)
  } catch (error) {
    console.error('Error training model:', error)
    trainingStatus.value = 'Error during training'
  }
}
</script>
