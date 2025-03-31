<script setup lang="ts">
import { watch, ref } from "vue";
import { runGAN } from "../store/api";
import { CODE_GAN_simple_v0 } from "../assets/codeSnippets";
import OnnxGANCanvas from "../components/OnnxGANCanvas.vue";
import CodeModal from "../components/CodeModal.vue";
import { useCreateToasts } from "../hooks/useCreateToasts";

type OnnxCanvasData =  {
  tensor: any[];
  dims: readonly number[];
  imgSize: number;
  channels: number;
} | null;

const { displayApiError } = useCreateToasts();

const { trigger, req } = runGAN();

const data = ref<OnnxCanvasData>(null);

watch(
  () => req.isErr,
  (val) => {
    if (val)
      displayApiError(req.errMsg)
  }
);

function generateGanSimpleV4(batchSize: number) {
  trigger({ batchSize, modelName: "GAN_simple_v4" }).then((res) => {
    if (!res?.data) return;
    data.value = {
      tensor: res?.data.tensor,
      dims: res?.data.dims,
      imgSize: res?.data.imgSize,
      channels: res?.data.outShape[1]
    }
  })
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">GAN_simple_v4 - MNIST</h2>
      <div class="ml-4">
        <OnnxGANCanvas
          :loading="req.loading"
          :data="data"
          @generate="generateGanSimpleV4"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="GAN_simple_v4 - MNIST"
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
