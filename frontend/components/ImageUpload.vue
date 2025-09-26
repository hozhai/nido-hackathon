<template>
  <div class="image-upload-component">
    <div class="upload-area" :class="{ 'drag-over': isDragOver }" @drop.prevent="handleDrop"
      @dragover.prevent="isDragOver = true" @dragleave.prevent="isDragOver = false">
      <input ref="fileInput" type="file" @change="handleFileChange" accept="image/*" multiple class="hidden">
      <div class="upload-content">
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7,10 12,15 17,10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        <p>Drop images here or click to select</p>
        <button @click="$refs.fileInput.click()" class="select-button">
          Select Images
        </button>
      </div>
    </div>

    <div v-if="uploadedFiles.length" class="file-list">
      <h3>Selected Files:</h3>
      <ul>
        <li v-for="(file, index) in uploadedFiles" :key="index" class="file-item">
          <span>{{ file.name }}</span>
          <button @click="removeFile(index)" class="remove-button">×</button>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  multiple: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['filesSelected'])

const isDragOver = ref(false)
const uploadedFiles = ref([])

const handleDrop = (event) => {
  isDragOver.value = false
  const files = Array.from(event.dataTransfer.files)
  processFiles(files)
}

const handleFileChange = (event) => {
  const files = Array.from(event.target.files)
  processFiles(files)
}

const processFiles = (files) => {
  const imageFiles = files.filter(file => file.type.startsWith('image/'))

  if (props.multiple) {
    uploadedFiles.value.push(...imageFiles)
  } else {
    uploadedFiles.value = imageFiles.slice(0, 1)
  }

  emit('filesSelected', uploadedFiles.value)
}

const removeFile = (index) => {
  uploadedFiles.value.splice(index, 1)
  emit('filesSelected', uploadedFiles.value)
}
</script>

<style scoped>
.upload-area {
  border: 2px dashed #cbd5e0;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}

.upload-area:hover,
.upload-area.drag-over {
  border-color: #3182ce;
  background-color: #ebf8ff;
}

.upload-icon {
  width: 48px;
  height: 48px;
  margin: 0 auto 1rem;
  color: #718096;
}

.select-button {
  background-color: #3182ce;
  color: white;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 1rem;
}

.select-button:hover {
  background-color: #2c5282;
}

.file-list {
  margin-top: 1rem;
}

.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background-color: #f7fafc;
  margin: 0.25rem 0;
  border-radius: 4px;
}

.remove-button {
  background: #e53e3e;
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  cursor: pointer;
}

.hidden {
  display: none;
}
</style>
