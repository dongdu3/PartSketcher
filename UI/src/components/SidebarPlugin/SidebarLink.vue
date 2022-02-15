<template>
  <li class="md-list-item">
    
    <router-link

      @click.native="changeModelType($attrs)"
      class="md-list-item-router md-list-item-container md-button-clean"

      v-bind="$attrs"
    >
      <div class="md-list-item-content md-ripple">
        <slot>
          <md-icon>{{ link.icon }}</md-icon>
          <p>{{ link.name}}</p>
        </slot>
      </div>
    </router-link>

  </li>
</template>


<script>
import {bus} from '../../../bus'
export default {
  inject: {
    autoClose: {
      default: true
    }
  },
  props: {
    link: {
      type: [String, Object],
      default: () => {
        return {
          name: "",
          path: "",
          icon: ""
        };
      }
    },
    tag: {
      type: String,
      default: "router-link"
    }
  },
  methods: {
    hideSidebar() {
      if (
        this.autoClose &&
        this.$sidebar &&
        this.$sidebar.showSidebar === true
      ) {
        this.$sidebar.displaySidebar(false);
      }
    },
    changeModelType(attrs){
      console.log("ask change model type",attrs.to)
      bus.$emit('update:modeltype')

    }
  }
};
</script>
<style></style>
