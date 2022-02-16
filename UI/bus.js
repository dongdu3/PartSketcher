import Vue from 'vue'
export var bus = new Vue({
    data:{
        current_drawing_tools:'tool-arrow',
        current_stroke_width: 3,
        current_model:{
            "version": "0.1.0",
            "comment": "Generated by MeshLab JSON Exporter",
            "id": 1,
            "name": "mesh",
            "vertices": [
              {
                "name": "position_buffer",
                "size": 3,
                "type": "float32",
                "normalized": false,
                "values": [
                  -0.25,
                  -0.25,
                  -0.25,
                  0.25,
                  -0.25,
                  -0.25,
                  -0.25,
                  0.25,
                  -0.25,
                  0.25,
                  0.25,
                  -0.25,
                  -0.25,
                  -0.25,
                  0.25,
                  0.25,
                  -0.25,
                  0.25,
                  -0.25,
                  0.25,
                  0.25,
                  0.25,
                  0.25,
                  0.25
                ]
              },
              {
                "name": "normal_buffer",
                "size": 3,
                "type": "float32",
                "normalized": false,
                "values": [
                  -0.57735,
                  -0.57735,
                  -0.57735,
                  0.333333,
                  -0.666667,
                  -0.666667,
                  -0.666667,
                  0.333333,
                  -0.666667,
                  0.666667,
                  0.666667,
                  -0.333333,
                  -0.666667,
                  -0.666667,
                  0.333333,
                  0.666667,
                  -0.333333,
                  0.666667,
                  -0.333333,
                  0.666667,
                  0.666667,
                  0.57735,
                  0.57735,
                  0.57735
                ]
              }
            ],
            "connectivity": [
              {
                "name": "triangles",
                "mode": "triangles_list",
                "indexed": true,
                "indexType": "uint32",
                "indices": [
                  2,
                  1,
                  0,
                  1,
                  2,
                  3,
                  4,
                  2,
                  0,
                  2,
                  4,
                  6,
                  1,
                  4,
                  0,
                  4,
                  1,
                  5,
                  6,
                  5,
                  7,
                  5,
                  6,
                  4,
                  3,
                  6,
                  7,
                  6,
                  3,
                  2,
                  5,
                  3,
                  7,
                  3,
                  5,
                  1
                ]
              }
            ],
            "mapping": [
              {
                "name": "standard",
                "primitives": "triangles",
                "attributes": [
                  {
                    "source": "position_buffer",
                    "semantic": "position",
                    "set": 0
                  },
                  {
                    "source": "normal_buffer",
                    "semantic": "normal",
                    "set": 0
                  }
                ]
              }
            ],
            "custom": null
          }
    },

})