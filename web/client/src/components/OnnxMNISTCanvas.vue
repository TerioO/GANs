<script setup lang="ts">
import { reactive, useTemplateRef, ref, watch, computed, onMounted } from "vue";
import type { IOnnxRequest } from "../types/api-types";
import Button from "primevue/button";
import InputNumber from "primevue/inputnumber";
import Slider from "primevue/slider";
import Dialog from "primevue/dialog";

interface Props {
  loading: boolean;
  data: IOnnxRequest["data"] | null;
}

const props = defineProps<Props>();

defineEmits<{
  (e: "generate", batchSize: string): void;
}>();

const canvas = useTemplateRef("canvas");
const canvasModal = useTemplateRef("canvasModal");

const form = reactive({
  batchSize: 1,
  canvasScale: 0
});
const modalVisible = ref(false);

const invalidBatchSize = computed(() => {
  return form.batchSize < 0 || form.batchSize > 64;
});

const canvasScale = computed(() => {
  return (1 + form.canvasScale / 20).toFixed(2);
});

onMounted(() => {
  if (props.data) createCanvasImageData();
});
watch(
  () => props.data,
  () => createCanvasImageData()
);
watch(
  () => canvasModal.value,
  () => {
    copyCanvasToModal();
  }
);

function toggleModal() {
  modalVisible.value = !modalVisible.value;
}

function copyCanvasToModal() {
  if (!canvas.value) return;
  if (!canvasModal.value) return;
  const ctx2 = canvasModal.value.getContext("2d");
  canvasModal.value.width = canvas.value.width;
  canvasModal.value.height = canvas.value.height;
  ctx2?.drawImage(canvas.value, 0, 0);
}

function createCanvasImageData() {
  if (!props.data) return;
  if (!canvas.value) return;

  const ctx = canvas.value.getContext("2d") as CanvasRenderingContext2D;
  const { tensor, dims } = props.data;

  const imgSize = 28;
  const batchSize = dims[0];
  const gridW = Math.ceil(Math.sqrt(batchSize));
  const gridH = Math.ceil(batchSize / gridW);

  const canvasW = imgSize * gridW;
  const canvasH = imgSize * gridH;
  canvas.value.width = canvasW;
  canvas.value.height = canvasW;
  const imageData = ctx.createImageData(canvasW, canvasH);

  for (let k = 0; k < batchSize; k++) {
    const gridX = (k % gridW) * imgSize;
    const gridY = Math.floor(k / gridW) * imgSize;
    for (let i = 0; i < imgSize; i++) {
      for (let j = 0; j < imgSize; j++) {
        const tensorIdx = k * imgSize * imgSize + i * imgSize + j;
        const value = tensor[tensorIdx];
        const imgDataIdx = ((gridY + i) * canvasW + (gridX + j)) * 4;

        imageData.data[imgDataIdx] = value;
        imageData.data[imgDataIdx + 1] = value;
        imageData.data[imgDataIdx + 2] = value;
        imageData.data[imgDataIdx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}
</script>
<template>
  <form class="flex flex-col">
    <div class="flex items-end gap-4">
      <div class="flex flex-col gap-1 relative">
        <label for="batch-size" class="text-slate-500 text-sm">Batch size</label>
        <InputNumber
          id="batch-size"
          v-model="form.batchSize"
          :invalid="invalidBatchSize"
        />
        <p v-if="invalidBatchSize" class="absolute -bottom-6 text-red-500 text-sm">
          Values between [1,64]
        </p>
      </div>
      <Button
        label="Generate"
        :disabled="loading || invalidBatchSize"
        @click="$emit('generate', form.batchSize.toString())"
      ></Button>
    </div>
    <div
      v-show="loading"
      class="animate-pulse w-[10rem] h-[10rem] bg-gray-300 mt-6"
    ></div>
    <div v-show="!loading && data" class="mt-9">
      <canvas
        ref="canvas"
        class="border-2 border-emerald-500 w-fit h-fit cursor-pointer"
        @click="toggleModal"
      ></canvas>
    </div>
    <Dialog
      v-model:visible="modalVisible"
      modal
      header="Image"
      :style="{ width: '90vw', height: '90vh' }"
      :pt="{
        content: {
          style: {
            height: '100%'
          }
        }
      }"
    >
      <div class="h-full flex flex-col">
        <div class="mb-4">
          <p class="mb-4 text-sm text-slate-500">Canvas Scale ({{ canvasScale }})</p>
          <Slider v-model="form.canvasScale" class="max-w-40" />
        </div>
        <div class="flex-1 overflow-auto flex items-center justify-center">
          <canvas
            ref="canvasModal"
            :style="{ transform: `scale(${canvasScale})` }"
          ></canvas>
        </div>
      </div>
    </Dialog>
  </form>
</template>
