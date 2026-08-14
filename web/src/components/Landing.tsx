import { useStore } from '../store'
import { ModelSearch } from './ModelSearch'

const EXAMPLES = [
  'Qwen/Qwen3-8B',
  'openai-community/gpt2',
  'meta-llama/Llama-3.1-8B',
  'mistralai/Mixtral-8x7B-Instruct-v0.1',
  'google/vit-base-patch16-224',
  'Qwen/Qwen2.5-VL-3B-Instruct',
]

export function Landing() {
  const loadModel = useStore((s) => s.loadModel)
  return (
    <div className="mm-landing">
      <h1 className="mm-wordmark">modelmap</h1>
      <p className="mm-tagline">
        Paste a Hugging Face model id. Get a living map of the network — no weights downloaded.
      </p>
      <ModelSearch big />
      <div className="mm-examples">
        {EXAMPLES.map((id) => (
          <button key={id} className="mm-example" onClick={() => void loadModel(id)}>
            {id}
          </button>
        ))}
      </div>
    </div>
  )
}
