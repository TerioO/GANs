<script setup lang="ts">
import AppHeader from './AppHeader.vue';
import AppFooter from './AppFooter.vue';
import { onMounted } from 'vue';
import { useApiStore } from '../store/apiStore';
import { useStore } from '../store/store';
import { useRouter } from 'vue-router';

const router = useRouter();
const { setServerStatus } = useStore();
const { getServerStatus } = useApiStore();

onMounted(() => {
    getServerStatus.trigger().then(() => {
        if (getServerStatus.axiosRes?.ok) {
            setServerStatus("ON")
            router.push("/")
        }
    });
})
</script>
<template>
    <div class="flex flex-col min-h-screen">
        <AppHeader />
        <div class="flex flex-1">
            <RouterView />
        </div>
        <AppFooter class="mt-auto" />
    </div>
</template>
