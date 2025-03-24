<script setup lang="ts">
interface Props {
    icon: string;
    iconClass: string;
    divClass: string;
}

defineProps<Props>();
</script>
<template>
    <div :class="divClass">
        <i :class="[
            'pi',
            icon,
            iconClass
        ]"></i>
    </div> 
</template>