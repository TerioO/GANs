<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue';

const timer = ref(0);
const interval = ref(0);

const minutes = computed(() => {
    return Math.floor(timer.value / 60);
})

onMounted(() => {
    interval.value = setInterval(() => {
        timer.value += 1;
    }, 1000)
})

onUnmounted(() => clearInterval(interval.value))
</script>
<template>
    <main class="container mx-auto flex flex-col flex-1 items-center justify-center px-4 py-8 text-center">
        <h1 class="text-xl mb-4">Waiting for server to turn on...</h1>
        <p>Render shuts down the server if it's not in use, this might take 1 minute or more.</p>
        <p>You can try a refresh if it takes too long</p>
        <div class="mt-4">Currently waiting for: {{ minutes ? `${minutes}m` : '' }} {{ timer % 60 }}s</div>
    </main>
</template>