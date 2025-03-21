<script setup lang="ts">
import { useApiStore } from '../store/apiStore';

const { getServerStatus } = useApiStore();

function temp(){
    getServerStatus.trigger();
}
</script>
<template>
    <main class="container mx-auto">
        <h1>HOME</h1>
        <button @click="temp">Get server status</button>
        <pre>{{ getServerStatus.loading }}</pre>
        <pre>{{ getServerStatus.isErr }}</pre>
        <pre>{{ getServerStatus.data }}</pre>
    </main>
</template>
