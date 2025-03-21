<script setup lang="ts">
import AppHeader from './AppHeader.vue';
import AppFooter from './AppFooter.vue';
</script>
<template>
    <div class="flex flex-col min-h-screen">
        <AppHeader />
        <div class="flex flex-1 p-4">
            <RouterView />
        </div>
        <AppFooter class="mt-auto"/>
    </div>
</template>
