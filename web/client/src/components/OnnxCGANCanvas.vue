<script setup lang="ts">
import { reactive, ref, computed, useTemplateRef } from "vue";
import Button from "primevue/button";
import InputNumber from "primevue/inputnumber";
import Slider from "primevue/slider";
import OnnxCanvas from "./OnnxCanvas.vue";

interface Props {
  loading: boolean;
  data: {
    tensor: any[];
    dims: readonly number[];
    imgSize: number;
    channels: number;
  } | null;
}

defineProps<Props>();

const emit = defineEmits<{
  (e: "generate", batchSize: number, label: number): void;
}>();

const onnxCanvas = useTemplateRef("onnxCanvas");

const form = reactive({
  batchSize: 1,
  label: 0,
  canvasScale: 0
});

const modalVisible = ref(false);

const invalidBatchSize = computed(() => {
  return form.batchSize < 0 || form.batchSize > 64;
});

const canvasScale = computed(() => {
  const newValue = (1 + form.canvasScale / 20).toFixed(2);
  const canvas = onnxCanvas.value?.canvasModal;
  if (canvas) canvas.style.transform = `scale(${canvasScale.value})`;
  return newValue;
});

function generate() {
  emit("generate", form.batchSize, form.label);
}

function toggleModal() {
  modalVisible.value = !modalVisible.value;
}
</script>
<template>
  <section class="flex flex-col">
    <div class="flex flex-col items-start lg:flex-row lg:items-end gap-4">
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
      <div class="flex flex-col gap-1 relative">
        <label for="label" class="text-slate-500 text-sm">Label</label>
        <InputNumber id="label" v-model="form.label" />
      </div>
      <Button
        label="Generate"
        :disabled="loading || invalidBatchSize"
        @click.prevent="generate"
      ></Button>
    </div>
    <OnnxCanvas
      ref="onnxCanvas"
      :modal-visible="modalVisible"
      :loading="loading"
      :data="data"
      @toggle-modal="toggleModal"
    >
      <template #modalControls>
        <div>
          <p class="mb-2 text-sm text-slate-500 text-center">
            Canvas Scale ({{ canvasScale }})
          </p>
          <Slider v-model="form.canvasScale" class="w-full" />
        </div>
        <div class="mb-7 mt-4 flex items-end gap-4">
          <div class="flex flex-col gap-1 relative">
            <label for="batch-size" class="text-slate-500 text-sm">Batch size</label>
            <InputNumber
              id="batch-size"
              size="small"
              v-model="form.batchSize"
              :invalid="invalidBatchSize"
            />
            <p v-if="invalidBatchSize" class="absolute -bottom-6 text-red-500 text-sm">
              Values between [1,64]
            </p>
          </div>
          <div class="flex flex-col gap-1 relative">
            <label for="label" class="text-slate-500 text-sm">Label</label>
            <InputNumber id="label" v-model="form.label" size="small" />
          </div>
          <Button
            label="Generate"
            size="small"
            :disabled="loading || invalidBatchSize"
            @click.prevent="generate"
          ></Button>
        </div>
      </template>
    </OnnxCanvas>
  </section>
</template>
