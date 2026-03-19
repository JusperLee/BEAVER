<script setup>
import { ref, onMounted } from 'vue';

const imageSeletorPaths = [
  "./images/qa.png",
  "./images/sum.png",
  "./images/code.png",
  "./images/few-shot.png"
];

const outputImagesPathsStr = [
  "./images/qa-big.png",
  "./images/sum-big.png",
  "./images/code-big.png",
  "./images/few-shot-big.png"
];

const outputImagesPaths = [];

let outputImagePath = ref("");
let indexSelected = ref(0);
let isLoading = ref(true);

const preloadImageSelector = () => {
  const promises = [];
      
  for (let i = 0; i < outputImagesPathsStr.length; i++) {
    const outputImg = new Image();
    outputImagesPaths[i] = outputImg;
    const outputPromise = new Promise((resolve) => {
        outputImg.onload = resolve;
    });
    outputImg.src = outputImagesPathsStr[i];
    promises.push(outputPromise);
  }

  return Promise.all(promises).then(() => {
    isLoading.value = false;
    console.log("preloadImageSelector finished");
  });
}

onMounted(() => {
    preloadImageSelector();
    handleChange(0);
});

const handleChange = (value) => {
  indexSelected.value = value;
  outputImagePath.value = outputImagesPaths[value].src;
};

</script>

<template>
  <div>
    <el-divider />

    <el-row justify="center">
      <h1 class="section-title">Different Task Visualization</h1>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="24" :md="22" :lg="20" :xl="18">
        <el-row justify="center" class="thumb-row">
          <el-col :span="6" class="thumb-col" v-for="(imageSeletorPath, index) in imageSeletorPaths" :key="index">
            <el-image
              class="image"
              :src="imageSeletorPath"
              style="aspect-ratio: 1;"
              fit="scale-down"
              @click="handleChange(index)"
              :class="{ 'selected-image': indexSelected === index, 'unselected-image': indexSelected !== index }"
            />
          </el-col>
        </el-row>
        <el-row justify="center" class="preview-row">
          <el-col :xs="24" :sm="24" :md="24" :lg="24" :xl="24" class="preview-col">
            <el-skeleton
              class="preview-skeleton-container"
              :loading="isLoading"
              animated
              :throttle="1000"
            >
              <template #template>
                <el-skeleton-item variant="image" class="preview-skeleton" />
              </template>
              <template #default>
                <img :src="outputImagePath" class="output-image" />
              </template>
            </el-skeleton>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>

.thumb-row {
  margin-top: 20px;
  justify-content: center;
}

.thumb-col {
  max-width: 140px;
}

.image {
  width: 100%;
  cursor: pointer;
}

.preview-row {
  margin-top: 16px;
}

.preview-col {
  display: flex;
  justify-content: center;
}

.preview-skeleton-container {
  width: min(900px, 100%);
}

.preview-skeleton {
  width: min(900px, 100%);
  aspect-ratio: 3 / 4;
  max-height: 90vh;
}

.output-image {
  width: min(900px, 100%);
  height: auto;
  max-height: 1680px;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

.image:hover{
  transition: none;
  box-shadow: 0px 0px 6px 0px #aaaaaa;
}

.selected-image{
  transition: 0.5s ease;
  box-shadow: 0px 0px 6px 0px #aaaaaa;
}


/* 未选中图像的样式，颜色变灰 */
.unselected-image {
  transition: 0.5s ease;
  opacity: 0.4;
}

</style>