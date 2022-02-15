<template>
  <div class="wrapper" style="height:780px; width:100% !important;">
    <side-bar>
    </side-bar>
    <top-navbar></top-navbar>
    <dashboard-content> </dashboard-content>
  </div>

</template>

<script>
//    -->
import TopNavbar from "./TopNavbar.vue";
import ContentFooter from "./ContentFooter.vue";
import DashboardContent from "./Content.vue";
import MobileMenu from "@/pages/Layout/MobileMenu.vue";
import FixedPlugin from "./Extra/FixedPlugin.vue";

export default {
  components: {
    TopNavbar,
    DashboardContent,
    ContentFooter,
    MobileMenu,
    FixedPlugin
  },
  data() {
    return {
    };
  }
};
</script>
