<script setup lang="ts">
import { ref } from "vue";
import Dialog from "primevue/dialog";
import Button from "primevue/button";

interface Props {
  code: string;
  header?: string;
  btnLabel?: string;
}

withDefaults(defineProps<Props>(), {
  header: "Code",
  btnLabel: "View code"
});

const modalVisible = ref(false);
</script>
<template>
  <div>
    <Button
      :label="btnLabel"
      @click="modalVisible = true"
      size="small"
      icon="pi pi-arrow-up-right-and-arrow-down-left-from-center"
      iconPos="right"
    />
    <Dialog
      v-model:visible="modalVisible"
      modal
      :header="header"
      :style="{ width: '90vw', height: '90vh' }"
      :pt="{
        content: {
          style: {
            height: '100%'
          }
        }
      }"
    >
      <highlightjs language="python" :code="code" ></highlightjs>
    </Dialog>
  </div>
</template>
