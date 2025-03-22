<script setup lang="ts">
import { watch } from 'vue';
import { useApiStore } from '../store/apiStore';
import { useToast } from 'primevue';
import OnnxMNISTCanvas from '../components/OnnxMNISTCanvas.vue';

const toast = useToast();

const { getGanSimpleV4 } = useApiStore();

watch(() => getGanSimpleV4.isErr, (newValue) => {
    if (newValue) toast.add({ severity: "error", summary: "API Error", detail: getGanSimpleV4.errMsg, life: 3000 })
})
</script>
<template>
    <section class="container mx-auto p-4">
        <section class="mt-4">
            <OnnxMNISTCanvas :loading="getGanSimpleV4.loading" :data="getGanSimpleV4.data"
                @generate="(batchSize) => getGanSimpleV4.trigger({ batchSize })" />
        </section>
    </section>
</template>