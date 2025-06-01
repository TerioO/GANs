<script setup lang="ts">
import { watch, ref } from "vue";
import { runGAN } from "../store/api";
import { CODE_DCGAN_MNIST_v1, CODE_DCGAN_Cats_v0 } from "../assets/codeSnippets";
import { useCreateToasts } from "../hooks/useCreateToasts";
import OnnxGANCanvas from "../components/OnnxGANCanvas.vue";
import CodeModal from "../components/CodeModal.vue";

type OnnxCanvasData = {
  tensor: any[];
  dims: readonly number[];
  imgSize: number;
  channels: number;
} | null;

const { displayApiError } = useCreateToasts();

const DCGAN_MNIST_v1 = runGAN();
const DCGAN_CATS_v0 = runGAN();

const data_DCGAN_MNIST_v1 = ref<OnnxCanvasData>(null);
const data_DCGAN_CATS_v0 = ref<OnnxCanvasData>(null);

watch(
  () => DCGAN_MNIST_v1.req.isErr,
  (val) => {
    if (val) displayApiError(DCGAN_MNIST_v1.req.errMsg);
  }
);

watch(
  () => DCGAN_CATS_v0.req.isErr,
  (val) => {
    if (val) displayApiError(DCGAN_CATS_v0.req.errMsg);
  }
);

function generateDCGAN_MNIST_v0(batchSize: number) {
  DCGAN_MNIST_v1.trigger({ batchSize, modelName: "DCGAN_MNIST_v1" }).then((res) => {
    if (!res) return;
    data_DCGAN_MNIST_v1.value = {
      tensor: res.data.tensor,
      dims: res.data.dims,
      imgSize: res.data.imgSize,
      channels: res.data.outShape[1]
    };
  });
}

function generateDCGAN_CATS_v0(batchSize: number) {
  DCGAN_CATS_v0.trigger({ batchSize, modelName: "DCGAN_Cats_v0" }).then((res) => {
    if (!res) return;
    data_DCGAN_CATS_v0.value = {
      tensor: res.data.tensor,
      dims: res.data.dims,
      imgSize: res.data.imgSize,
      channels: res.data.outShape[1]
    };
  });
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">DCGAN_MNIST_v1</h2>
      <div class="ml-4">
        <OnnxGANCanvas
          :loading="DCGAN_MNIST_v1.req.loading"
          :data="data_DCGAN_MNIST_v1"
          @generate="generateDCGAN_MNIST_v0"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="DCGAN_MNIST_v1"
          :code="CODE_DCGAN_MNIST_v1"
        />
        <div class="flex flex-col gap-4 mt-6">
          <p>Epochs: 125</p>
          <p>Learning rates: 0.0002</p>
        </div>
      </div>
    </section>
    <section class="mt-9">
      <h2 class="text-xl font-bold mb-4">DCGAN_CATS_v0</h2>
      <div class="ml-4">
        <OnnxGANCanvas
          :loading="DCGAN_CATS_v0.req.loading"
          :data="data_DCGAN_CATS_v0"
          @generate="generateDCGAN_CATS_v0"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="DCGAN_CATS_v0"
          :code="CODE_DCGAN_Cats_v0"
        />
        <div class="flex flex-col gap-4 mt-6">
          <p>GAN epochs = 400</p>
          <p>
            The Generator & Discriminator follow the architecture from the DCGAN paper
            adapted to RGB images
          </p>
          <p>
            Models where trained with a batch_size=64, Generator LR=2e-4 and Discriminator
            LR=1e-4, both having used Adam optimizer with betas=(0.5, 0.999)
          </p>
        </div>
      </div>
    </section>
  </section>
</template>
