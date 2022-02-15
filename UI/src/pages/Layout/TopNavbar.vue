<template>
  <md-toolbar md-elevation="0" class="md-transparent" style="overflow:hidden;border-bottom:1px #252525 solid;background-color:#010101 !important;height:70px;padding:0 0;">
    <div class="md-toolbar-row">
      <div class="md-toolbar-section-start">
        <tabs
          :tabs="tabs"
          :currentTab="currentTab"
          :wrapper-class="'default-tabs'"
          :tab-class="'default-tabs__item'"
          :tab-active-class="'default-tabs__item_active'"
          :line-class="'default-tabs__active-line'"
          @onClick="handleClick"
        >
        </tabs>
      </div>
      <div class="md-toolbar-section-end">
        <md-list>
          
          <button class="download-button" v-on:click="save_current_model">
            <i class="material-icons" style="margin-left:10px !important;margin-top:8px;" >cloud_download</i>
            <div style="margin-top:-2px;margin-left:15px;"> Download Model </div>
          </button>
          <!--
          <md-list-item v-on:click="askCopyCurrentLayer">
              <i class="material-icons">content_paste</i>
              <p class="hidden-lg hidden-md">change type</p>
          </md-list-item>
          
          <md-list-item v-on:click="askChangeViewType">
              <i class="material-icons">edit</i>
              <p class="hidden-lg hidden-md">change type</p>
          </md-list-item>
          
          <md-list-item v-on:click="askAssembly" >
              <i class="material-icons">dynamic_feed</i>
              <p class="hidden-lg hidden-md">assemble</p>
          </md-list-item>

          <md-list-item v-on:click="add_new_layer" >
              <i class="material-icons">add</i>
              <p class="hidden-lg hidden-md">New Layer</p>
          </md-list-item>
          
          <md-list-item v-on:click="clear_current_layer">
              <i class="material-icons">layers_clear</i>
              <p class="hidden-lg hidden-md">Clear Layer</p>
          </md-list-item>
          
          <md-list-item v-on:click="delete_current_layer">
              <i class="material-icons">delete</i>
              <p class="hidden-lg hidden-md">Delete Layer</p>
          </md-list-item>

          <md-list-item v-on:click="save_current_model">
              <i class="material-icons">save</i>
              <p class="hidden-lg hidden-md">Save File</p>
          </md-list-item>

          <md-list-item >
              <i class="material-icons">folder</i>
              <p class="hidden-lg hidden-md">Load File</p>
          </md-list-item>
          -->
          
        </md-list>
      </div>
    </div>
  </md-toolbar>


</template>

<script>


  
import {bus} from '../../../bus'
import Tabs from 'vue-tabs-with-active-line';

const TABS = [{
  title: 'Sketch',
  value: 'Sketch',
}, {
  title: 'Preview',
  value: 'Preview',
}];


export default {
  components: {
    Tabs,
  },
  data() {
    return {
      selectedEmployee: null,
      employees: [
      ],
      tabs: TABS,
      currentTab: 'Sketch',
      displayType:[{
          'name':'Sketch',
          'activated':true,
        },{
          'name':'Preview',
          'activated':false,
        },
      ],
    };
  },
  methods: {
    askCopyCurrentLayer(){
      bus.$emit("update:askPasteCurrentLayer")
    },
    askChangeViewType(){
      bus.$emit("update:askChangeViewType")
    },
    askAssembly(){
      bus.$emit("update:askAssembly")
    },
    toggleSidebar() {
      this.$sidebar.displaySidebar(!this.$sidebar.showSidebar);
    },
    clear_current_layer(){
        //console.log("clear current layer")
        bus.$emit("update:clearChairLayer")
    },
    add_new_layer(){
        //console.log("clear current layer")
        bus.$emit("update:addChairLayer")
    },
    delete_current_layer(){
        bus.$emit("update:deleteChairLayer")
    },
    save_current_model(){
        bus.$emit("save:current:model")
    },
    handleClick(newTab) {
      this.currentTab = newTab;
      bus.$emit('update:changeviewtype',newTab)
    },
  },
  mounted() {
    bus.$on('hack:header',()=>{
      this.$data.currentTab= 'Preview'
    })
  }

};
</script>

<style lang="scss">
.download-button{

  height: 40px;
  width: 200px;
  border-radius: 5px;
  background-color: #116df7;
  font-size: 16px;
  color: white;
  text-align: justify;
  line-height: 45px;
  transition-property:opacity;


  border-width: 0px;
  //margin-top:;
  display: flex; 
  flex-direction: row;
  &:hover {
    opacity: 0.7;
    transition-duration:.2s;
    transition-timing-function:cubic-bezier(.02,.01,.05,1);
  }

}
.default-tabs {
  position: relative;
  left:-90px;
  margin: 0 auto;
  &__item {
    display: inline-block;
    margin: 0 25px;
    height: 70px;
    padding: 10px;
    padding-bottom: 8px;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 0.8px;
    color: gray;
    text-decoration: none;
    border: none;
    background-color: transparent;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: all 0.25s;
    &_active {
      color: white;
      border-bottom: 4px solid #116df7;
    }
    &:hover {
      border-bottom: 4px solid #116df7;
      color: white
    }
    &:focus {
      outline: none;
      border-bottom: 4px solid #116df7;
      color: white
    }
    &:first-child {
      margin-left: 0;
    }
    &:last-child {
      margin-right: 0;
    }
  }
  &__active-line {
    position: absolute;
    bottom: 0;
    left: 0;
    height: 4px;
    background-color: #116df7;
    transition: transform 0.4s ease, width 0.4s ease;
  }
}
</style>
