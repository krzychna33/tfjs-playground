import * as tf from '@tensorflow/tfjs-node'
import {cos, Tensor3D, Tensor4D} from "@tensorflow/tfjs-node";
import * as fs from 'node:fs/promises'
import sharp from 'sharp'

console.log(tf.version.tfjs)

const MODEL_PATH =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";

let movenet: tf.GraphModel;

async function loadModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, {fromTFHub: true});

  const imgBuffer = await fs.readFile('src/img.png')
  const inputTensor = tf.node.decodePng(imgBuffer, 3)

  const h = 460;
  const w = 460;

  let cropStartPoint = [128, 256, 0]
  let cropSize = [460, 460, 3]

  let croppedTensor = tf.slice(inputTensor, cropStartPoint, cropSize);
  let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt()


  let croppedImg = await tf.node.encodePng(croppedTensor)


  const tensorOutput = movenet.predict(tf.expandDims(resizedTensor)) as Tensor4D
  const arrayedOutput = await tensorOutput.array()

  const svgElipses = arrayedOutput[0][0].map((point: number[]) => {
    const y = point[0] * h;
    const x = point[1] * w;
    return `<ellipse ry="5" rx="5" id="svg_8" cy="${y}" cx="${x}" stroke-width="0" stroke="#050505" fill="#ff0000"/>`
  }).join(' ')


  const svg = `<svg width="460" height="460" xmlns="http://www.w3.org/2000/svg" stroke="null">
 <g stroke="null">
  <title stroke="null">Layer 1</title>
  ${svgElipses}
 </g>

</svg>`

  const svgOverlay = Buffer.from(svg);

  await sharp(croppedImg)
    .composite([
      {
        input: svgOverlay,
        top: 0,
        left: 0
      }
    ])
    .toFile('src/img2.png')



  inputTensor.dispose()
  croppedTensor.dispose()
  resizedTensor.dispose()
  tensorOutput.dispose()


}

loadModel()