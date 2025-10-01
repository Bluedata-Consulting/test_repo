export function encodeWav16kMono(int16: Int16Array, sampleRate = 16000): Blob {
  const buffer = new ArrayBuffer(44 + int16.length * 2)
  const view = new DataView(buffer)
  // RIFF header
  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + int16.length * 2, true)
  writeString(view, 8, 'WAVE')
  // fmt chunk
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // PCM
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  // data chunk
  writeString(view, 36, 'data')
  view.setUint32(40, int16.length * 2, true)
  let offset = 44
  for (let i = 0; i < int16.length; i++, offset += 2) view.setInt16(offset, int16[i], true)
  return new Blob([view], { type: 'audio/wav' })
}

function writeString(view: DataView, offset: number, s: string) {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
}


