<script setup lang="ts">
import { watch, reactive } from "vue";
import { useApiStore } from "../store/apiStore";
import { useToast } from "primevue";
import { CODE_DCGAN_MNIST_v0, CODE_DCGAN_Cats_v0 } from "../assets/codeSnippets";
import type { IOnnxRequest } from "../types/api-types";
import OnnxGANCanvas from "../components/OnnxGANCanvas.vue";
import CodeModal from "../components/CodeModal.vue";

interface DCGAN {
  loading: boolean;
  data: IOnnxRequest["res"] | null;
}

const toast = useToast();

const { runGAN } = useApiStore();

const DCGAN_MNIST_v0: DCGAN = reactive({
  loading: false,
  data: null
});
const DCGAN_CATS_v0: DCGAN = reactive({
  loading: false,
  data: null
});

watch(
  () => runGAN.isErr,
  (newValue) => {
    if (newValue)
      toast.add({
        severity: "error",
        summary: "API Error",
        detail: runGAN.errMsg,
        life: 3000
      });
  }
);

function generateDCGAN_MNIST_v0(batchSize: string) {
  DCGAN_MNIST_v0.loading = true;
  runGAN
    .trigger({ batchSize, modelName: "DCGAN_MNIST_v0" })
    .then((res) => {
      if (!res?.value) return;
      DCGAN_MNIST_v0.data = res.value.data;
    })
    .finally(() => (DCGAN_MNIST_v0.loading = false));
}

function generateDCGAN_CATS_v0(batchSize: string) {
  DCGAN_CATS_v0.loading = true;
  runGAN
    .trigger({ batchSize, modelName: "DCGAN_Cats_v0" })
    .then((res) => {
      if (!res?.value) return;
      DCGAN_CATS_v0.data = res.value.data;
    })
    .finally(() => (DCGAN_CATS_v0.loading = false));
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">DCGAN_MNIST_v0</h2>
      <div class="ml-4">
        <OnnxGANCanvas
          :loading="DCGAN_MNIST_v0.loading"
          :data="DCGAN_MNIST_v0.data"
          @generate="generateDCGAN_MNIST_v0"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="Code"
          :code="CODE_DCGAN_MNIST_v0"
        />
        <div class="flex flex-col gap-4 mt-6">
          <p>GAN epochs = 400</p>
          <p>
            The Generator & Discriminator follow the architecture from the DCGAN paper
            adapted to MNIST dataset
          </p>
          <p>
            Models where trained with a batch_size=128, Generator LR=2e-4 and
            Discriminator LR=1e-4, both having used Adam optimizer with betas=(0.5, 0.999)
          </p>
        </div>
      </div>
    </section>
    <section class="mt-9">
      <h2 class="text-xl font-bold mb-4">DCGAN_CATS_v0</h2>
      <div class="ml-4">
        <OnnxGANCanvas
          :loading="DCGAN_CATS_v0.loading"
          :data="DCGAN_CATS_v0.data"
          @generate="generateDCGAN_CATS_v0"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="Code"
          :code="CODE_DCGAN_Cats_v0"
        />
        <div class="flex flex-col gap-4 mt-6">
          <p>GAN epochs = 400</p>
          <p>
            The Generator & Discriminator follow the architecture from the DCGAN paper
            adapted to RGB images
          </p>
          <p>
            Models where trained with a batch_size=64, Generator LR=2e-4 and
            Discriminator LR=1e-4, both having used Adam optimizer with betas=(0.5, 0.999)
          </p>
        </div>
      </div>
    </section>
  </section>
</template>
