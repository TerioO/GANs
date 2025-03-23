import "./style.css";
import "primeicons/primeicons.css";
import App from "./App.vue";
import { createApp } from "vue";
import { createPinia } from "pinia";
import { router } from "./router";
import PrimeVue from "primevue/config";
import Aura from "@primeuix/themes/aura";
import ToastService from "primevue/toastservice";

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
  .mount("#app");
