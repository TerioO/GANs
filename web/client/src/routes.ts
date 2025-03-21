import type { RouteRecordRaw } from "vue-router";
import HomeView from "./views/HomeView.vue";
import _404View from "./views/_404View.vue";

export const routes: RouteRecordRaw[] = [
    { path: "/", name: "Home", component: HomeView },
    { path: "/:pathMatch(.*)*", name: "404 - Not Found", component: _404View }
];
