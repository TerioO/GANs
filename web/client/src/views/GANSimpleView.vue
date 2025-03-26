<script setup lang="ts">
import { watch } from "vue";
import { runGAN } from "../store/api";
import { useToast } from "primevue";
import { CODE_GAN_simple_v0 } from "../assets/codeSnippets";
import OnnxGANCanvas from "../components/OnnxGANCanvas.vue";
import CodeModal from "../components/CodeModal.vue";

const toast = useToast();

const { trigger, req } = runGAN();

watch(
  () => req.isErr,
  (newValue) => {
    if (newValue)
      toast.add({
        severity: "error",
        summary: "API Error",
        detail: req.errMsg,
        life: 3000
      });
  }
);

function generateGanSimpleV4(batchSize: string) {
  trigger({ batchSize, modelName: "GAN_simple_v4" })
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">GAN_simple_v4 - MNIST</h2>
      <div class="ml-4">
        <OnnxGANCanvas
          :loading="req.loading"
          :data="req.data"
          @generate="generateGanSimpleV4"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="Code"
          :code="CODE_GAN_simple_v0"
        />
        <div class="flex flex-col gap-4 mt-6">
          <p>GAN epochs = 620</p>
          <p>
            The Generator is made of 3 Linear layers with with ReLU activations,
            BatchNorm1d on every layer except the output layer and output is passed
            through Tanh
          </p>
          <p>
            The Discriminator is made of 3 Linear layers, output is passed through Sigmoid
            and BatchNorm1d is performed one very layer except the last.
          </p>
          <p>
            Models where trained with a batch_size=128, Generator LR=2e-4 and
            Discriminator LR=1e-4, both having used Adam optimizer
          </p>
        </div>
      </div>
    </section>
  </section>
</template>
