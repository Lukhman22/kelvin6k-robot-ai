import { useState, useRef, useEffect } from 'react';

export function IndependentMicTest() {
  const [status, setStatus] = useState<string>('Idle');
  const [peak, setPeak] = useState<number>(0);
  const [trackInfo, setTrackInfo] = useState<any>(null);
  const [permissionState, setPermissionState] = useState<string>('unknown');
  
  // Recording state
  const [blobInfo, setBlobInfo] = useState<{ size: number, type: string, duration: number } | null>(null);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<number | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    navigator.permissions?.query({ name: 'microphone' as PermissionName })
      .then(res => {
        setPermissionState(res.state);
        res.addEventListener('change', () => setPermissionState(res.state));
      })
      .catch(() => setPermissionState('not supported'));
  }, []);

  const getMimeType = () => {
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    return candidates.find(t => MediaRecorder.isTypeSupported?.(t)) || "";
  };

  const startTest = async () => {
    stopTest();
    setStatus('Requesting getUserMedia...');
    setPeak(0);
    setTrackInfo(null);
    setBlobInfo(null);
    if (playbackUrl) {
      URL.revokeObjectURL(playbackUrl);
      setPlaybackUrl(null);
    }
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const track = stream.getAudioTracks()[0];
      setTrackInfo({
        label: track.label,
        muted: track.muted,
        readyState: track.readyState,
        enabled: track.enabled,
      });

      const updateTrackInfo = () => {
        setTrackInfo({
          label: track.label,
          muted: track.muted,
          readyState: track.readyState,
          enabled: track.enabled,
        });
      };
      track.addEventListener('mute', updateTrackInfo);
      track.addEventListener('unmute', updateTrackInfo);

      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioContextClass();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const buffer = new Float32Array(analyser.fftSize);
      let localPeak = 0;

      timerRef.current = window.setInterval(() => {
        analyser.getFloatTimeDomainData(buffer);
        let currentTickPeak = 0;
        for (let i = 0; i < buffer.length; i++) {
          if (Math.abs(buffer[i]) > currentTickPeak) {
            currentTickPeak = Math.abs(buffer[i]);
          }
        }
        if (currentTickPeak > localPeak) {
          localPeak = currentTickPeak;
          setPeak(localPeak);
        }
        updateTrackInfo();
      }, 100);

      // Start recording
      const mimeType = getMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const duration = Date.now() - startTimeRef.current;
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || mimeType || 'audio/webm' });
        setBlobInfo({ size: blob.size, type: blob.type, duration });
        
        const url = URL.createObjectURL(blob);
        setPlaybackUrl(url);
        stopResources();
      };

      startTimeRef.current = Date.now();
      recorder.start();
      setStatus('Recording 5 seconds of audio...');

      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
          setStatus('Recording complete.');
        }
      }, 5000);

    } catch (e: any) {
      setStatus(`Error: ${e.name} - ${e.message}`);
    }
  };

  const stopResources = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  };

  const stopTest = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    } else {
      stopResources();
    }
    setStatus('Stopped');
  };

  return (
    <div style={{ margin: '10px', padding: '15px', border: '2px solid #d32f2f', backgroundColor: '#ffebee', color: '#000', borderRadius: '4px' }}>
      <h4 style={{ margin: '0 0 10px 0' }}>Independent Microphone Diagnostic (Phase 2)</h4>
      <p style={{ margin: '0 0 15px 0' }}>Bypasses all app logic. Tests getUserMedia and MediaRecorder directly.</p>
      
      <div style={{ marginBottom: '15px' }}>
        <button onClick={startTest} style={{ padding: '8px 16px', marginRight: '10px', cursor: 'pointer', background: '#d32f2f', color: '#fff', border: 'none', borderRadius: '4px' }}>START 5s TEST</button>
        <button onClick={stopTest} style={{ padding: '8px 16px', cursor: 'pointer' }}>STOP</button>
      </div>
      
      <div style={{ background: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #ccc' }}>
        <p style={{ margin: '4px 0' }}><strong>Status:</strong> {status}</p>
        <p style={{ margin: '4px 0' }}><strong>Permission:</strong> {permissionState}</p>
        
        {trackInfo && (
          <div style={{ marginTop: '10px' }}>
            <p style={{ margin: '4px 0' }}><strong>Device:</strong> {trackInfo.label}</p>
            <p style={{ margin: '4px 0' }}><strong>Track state:</strong> {trackInfo.readyState} | <strong>Enabled:</strong> {String(trackInfo.enabled)} | <strong>Muted:</strong> {String(trackInfo.muted)}</p>
          </div>
        )}

        <div style={{ margin: '15px 0' }}>
          <strong>Live Peak:</strong> {peak.toFixed(4)}
          <div style={{ width: '100%', height: '20px', background: '#eee', marginTop: '5px' }}>
            <div style={{ width: `${Math.min(100, peak * 200)}%`, height: '100%', background: '#4caf50', transition: 'width 0.1s' }} />
          </div>
        </div>

        {blobInfo && (
          <div style={{ marginTop: '15px', padding: '10px', background: '#e8f5e9', border: '1px solid #4caf50', borderRadius: '4px' }}>
            <strong>Recording Results:</strong>
            <ul style={{ margin: '5px 0 10px 15px', padding: 0 }}>
              <li>Size: {blobInfo.size} bytes</li>
              <li>MIME Type: {blobInfo.type}</li>
              <li>Duration: {(blobInfo.duration / 1000).toFixed(2)}s</li>
            </ul>
            {playbackUrl && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <audio controls src={playbackUrl} />
                <a href={playbackUrl} download={`mic_test_${Date.now()}.webm`} style={{ color: '#1976d2', textDecoration: 'none', fontWeight: 'bold' }}>Download Recording</a>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
