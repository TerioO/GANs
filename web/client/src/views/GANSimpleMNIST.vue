<script setup lang="ts">
import { watch } from "vue";
import { useApiStore } from "../store/apiStore";
import { useToast } from "primevue";
import { CODE_GAN_simple_v0 } from "../assets/codeSnippets";
import OnnxGANCanvas from "../components/OnnxGANCanvas.vue";
import CodeModal from "../components/CodeModal.vue";

const toast = useToast();

const { getGanSimpleV4 } = useApiStore();

watch(
  () => getGanSimpleV4.isErr,
  (newValue) => {
    if (newValue)
      toast.add({
        severity: "error",
        summary: "API Error",
        detail: getGanSimpleV4.errMsg,
        life: 3000
      });
  }
);

function generateGanSimpleV4(batchSize: string) {
  getGanSimpleV4.trigger({ batchSize, modelName: "GAN_simple_v4" });
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">GAN_simple_v4</h2>
      <OnnxGANCanvas
        class="ml-4"
        :loading="getGanSimpleV4.loading"
        :data="getGanSimpleV4.data"
        @generate="generateGanSimpleV4"
      />
      <CodeModal class="mt-9 ml-4" header="Code" :code="CODE_GAN_simple_v0" />
    </section>
  </section>
</template>
