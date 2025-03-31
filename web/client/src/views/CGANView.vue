<script setup lang="ts">
import { watch, ref } from "vue";
import { runCGAN } from "../store/api";
import { useCreateToasts } from "../hooks/useCreateToasts";
import OnnxCGANCanvas from "../components/OnnxCGANCanvas.vue";

type OnnxCanvasData = {
  tensor: any[];
  dims: readonly number[];
  imgSize: number;
  channels: number;
} | null;

const { displayApiError } = useCreateToasts();

const CDCGAN_MNIST_v0 = runCGAN();
const CDCGAN_Cats_v0 = runCGAN();

const data_CDCGAN_MNIST_v0 = ref<OnnxCanvasData>(null);
const data_CDCGAN_Cats_v0 = ref<OnnxCanvasData>(null);

watch(
  () => CDCGAN_MNIST_v0.req.isErr,
  (val) => {
    if (val) displayApiError(CDCGAN_MNIST_v0.req.errMsg);
  }
);

function generateCDGAN_MNIST_v0(batchSize: number, label: number) {
  CDCGAN_MNIST_v0.trigger({ batchSize, label, modelName: "CDCGAN_MNIST_v0" }).then(
    (res) => {
      if (!res) return;
      data_CDCGAN_MNIST_v0.value = {
        tensor: res.data.tensor,
        dims: res.data.dims,
        imgSize: res.data.imgSize,
        channels: res.data.outShape[1]
      };
    }
  );
}

function generateCDCGAN_Cats_v0(batchSize: number, label: number) {
  CDCGAN_Cats_v0.trigger({ batchSize, label, modelName: "CDCGAN_Cats_v0" }).then(
    (res) => {
      if (!res) return;
      data_CDCGAN_Cats_v0.value = {
        tensor: res.data.tensor,
        dims: res.data.dims,
        imgSize: res.data.imgSize,
        channels: res.data.outShape[1]
      };
    }
  );
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <p>
        <span class="font-bold">Label</span> input takes a number (int) represeting the
        class of the image you want to generate.
      </p>
      <p>
        Datasets can have multiple values for label, MNIST for example has 10 (0 to 9)
      </p>
      <p>
        If you insert a value bellow or above the dataset's existing classes, the API will
        return the labels in order (0 to n) if batch_size allows it
      </p>
    </section>
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">CDCGAN_MNIST_v0</h2>
      <div class="ml-4">
        <OnnxCGANCanvas
          :loading="CDCGAN_MNIST_v0.req.loading"
          :data="data_CDCGAN_MNIST_v0"
          @generate="generateCDGAN_MNIST_v0"
        />
        <div class="flex flex-col mt-6">
          <p>Labels: 0 -> 10</p>
          <p>Epochs: 200</p>
        </div>
      </div>
    </section>
    <section class="mt-9">
      <h2 class="text-xl font-bold mb-4">CDCGAN_Cats_v0</h2>
      <div class="ml-4">
        <OnnxCGANCanvas
          :loading="CDCGAN_Cats_v0.req.loading"
          :data="data_CDCGAN_Cats_v0"
          @generate="generateCDCGAN_Cats_v0"
        />
        <div class="flex flex-col mt-6">
          <p>Labels: 0, 1</p>
          <p>Epochs: 550</p>
        </div>
      </div>
    </section>
  </section>
</template>
