import "./style.css";
import "highlight.js/styles/github-dark.css";
import "primeicons/primeicons.css";
import App from "./App.vue";
import { createApp } from "vue";
import { createPinia } from "pinia";
import { router } from "./router";
import PrimeVue from "primevue/config";
import Aura from "@primeuix/themes/aura";
import ToastService from "primevue/toastservice";
import hljs from 'highlight.js/lib/core';
import python from "highlight.js/lib/languages/python";
import hljsVuePlugin from "@highlightjs/vue-plugin";

hljs.registerLanguage("python", python);

const app = createApp(App);
const pinia = createPinia();

app
  .use(pinia)
  .use(router)
  .use(PrimeVue, {
    theme: {
      preset: Aura,
      options: {
        darkModeSelector: "nan"
      }
    }
  })
  .use(ToastService)
  .use(hljsVuePlugin)
  .mount("#app");
