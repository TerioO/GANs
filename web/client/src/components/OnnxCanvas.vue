<script setup lang="ts">
import { ref, useTemplateRef, watch, onMounted } from "vue";
import Dialog from "primevue/dialog";

interface Props {
  loading: boolean;
  modalVisible: boolean;
  data: {
    tensor: any[];
    dims: readonly number[];
    imgSize: number;
    channels: number;
  } | null;
}

const props = defineProps<Props>();

defineEmits(["toggleModal"]);

const canvas = useTemplateRef("canvas");
const canvasModal = useTemplateRef("canvasModal");

const modalVis = ref(props.modalVisible);

watch(
  () => props.modalVisible,
  (newValue) => (modalVis.value = newValue)
);

onMounted(() => {
  if (props.data) createCanvasImageData();
});
watch(
  () => props.data,
  () => {
    createCanvasImageData();
    if (canvasModal.value) copyCanvasToModal();
  }
);
watch(
  () => canvasModal.value,
  () => {
    copyCanvasToModal();
  }
);

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

  const imgSize = props.data.imgSize;
  const batchSize = dims[0];
  const channels = props.data.channels;
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
        const tensorIdx = k * channels * imgSize * imgSize + i * imgSize + j;
        const imgDataIdx = ((gridY + i) * canvasW + (gridX + j)) * 4;

        const values =
          channels <= 1
            ? [tensor[tensorIdx], tensor[tensorIdx], tensor[tensorIdx]]
            : [
                tensor[tensorIdx],
                tensor[tensorIdx + imgSize * imgSize],
                tensor[tensorIdx + 2 * imgSize * imgSize]
              ];

        imageData.data[imgDataIdx] = values[0];
        imageData.data[imgDataIdx + 1] = values[1];
        imageData.data[imgDataIdx + 2] = values[2];
        imageData.data[imgDataIdx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

defineExpose({ canvas, canvasModal });
</script>
<template>
  <div>
    <div
      v-show="loading"
      class="animate-pulse w-[10rem] h-[10rem] bg-gray-300 mt-6"
    ></div>
    <div v-show="!loading && data" class="mt-9 relative overflow-x-auto max-w-[80vw]">
      <canvas
        ref="canvas"
        class="cursor-pointer"
        @click="$emit('toggleModal')"
      ></canvas>
    </div>
    <Dialog
      v-model:visible="modalVis"
      modal
      header="Image"
      :style="{ width: '90vw', height: '90vh' }"
      :pt="{
        header: {
          style: {
            paddingBottom: '0'
          }
        },
        content: {
          style: {
            height: '100%'
          }
        }
      }"
    >
      <div class="h-full flex flex-col">
        <slot name="modalControls"></slot>
        <div class="flex-1 overflow-auto flex items-center justify-center">
          <div
            v-show="loading"
            class="animate-pulse h-1/2 aspect-square bg-gray-300 mt-6"
          ></div>
          <div v-show="!loading && data">
            <canvas ref="canvasModal"></canvas>
          </div>
        </div>
      </div>
    </Dialog>
  </div>
</template>
