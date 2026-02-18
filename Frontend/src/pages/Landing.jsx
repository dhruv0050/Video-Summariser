import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import VideoChatBot from '../components/VideoChatBot';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const Landing = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [videoUrl, setVideoUrl] = useState('');
  const [videoName, setVideoName] = useState('');
  const [jobId, setJobId] = useState('');
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [polling, setPolling] = useState(false);
  const [viewingPastReport, setViewingPastReport] = useState(false);
  const [videoSource, setVideoSource] = useState('auto'); // 'auto', 'drive', 'youtube', 'upload'
  const [uploadedFile, setUploadedFile] = useState(null);

  // Detect video source from URL
  const detectVideoSource = (url) => {
    if (!url) return 'auto';
    const youtubePattern = /(?:youtube\.com|youtu\.be)/;
    const drivePattern = /drive\.google\.com/;
    
    if (youtubePattern.test(url)) return 'youtube';
    if (drivePattern.test(url)) return 'drive';
    return 'auto';
  };

  const canSubmit = useMemo(() => {
    if (videoSource === 'upload') {
      return uploadedFile !== null;
    }
    return videoUrl.trim().length > 0;
  }, [videoUrl, videoSource, uploadedFile]);

  // Check if we're viewing a past report from URL params
  useEffect(() => {
    const queryJobId = searchParams.get('jobId');
    if (queryJobId) {
      setViewingPastReport(true);
      setJobId(queryJobId);
      fetchResults(queryJobId);
    }
  }, [searchParams]);

  useEffect(() => {
    let interval;
    if (jobId && polling && !viewingPastReport) {
      interval = setInterval(() => pollStatus(jobId), 5000);
      pollStatus(jobId); // immediate poll
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, polling, viewingPastReport]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setStatus('pending');
    setProgress(0);
    setJobId('');

    // Handle file upload
    if (videoSource === 'upload' || uploadedFile) {
      if (!uploadedFile) {
        setError('Please select a video file to upload');
        setStatus('idle');
        return;
      }

      try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        if (videoName.trim()) {
          formData.append('video_name', videoName.trim());
        }

        const resp = await fetch(`${API_BASE}/api/videos/process-upload`, {
          method: 'POST',
          body: formData,
        });

        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || 'Failed to start processing');
        }

        const data = await resp.json();
        setJobId(data.job_id);
        setStatus(data.status || 'pending');
        setProgress(data.progress || 0);
        setPolling(true);
        setUploadedFile(null); // Reset file input
        return;
      } catch (err) {
        setError(err.message || 'Something went wrong');
        setStatus('failed');
        return;
      }
    }

    // Handle URL-based processing
    const url = videoUrl.trim();
    const detectedSource = detectVideoSource(url);
    const source = videoSource === 'auto' ? detectedSource : videoSource;

    // Validate URL based on source
    if (source === 'auto' || (!detectedSource.includes('youtube') && !detectedSource.includes('drive'))) {
      setError('Please enter a valid Google Drive or YouTube URL');
      setStatus('idle');
      return;
    }

    try {
      let endpoint, payload;
      
      if (source === 'youtube') {
        endpoint = `${API_BASE}/api/videos/process-youtube`;
        payload = {
          youtube_url: url,
          video_name: videoName.trim() || 'Untitled Video',
        };
      } else {
        endpoint = `${API_BASE}/api/videos/process`;
        payload = {
          drive_video_url: url,
          video_name: videoName.trim() || 'Untitled Video',
        };
      }

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || 'Failed to start processing');
      }

      const data = await resp.json();
      setJobId(data.job_id);
      setStatus(data.status || 'pending');
      setProgress(data.progress || 0);
      setPolling(true);
    } catch (err) {
      setError(err.message || 'Something went wrong');
      setStatus('failed');
    }
  };

  const pollStatus = async (id) => {
    try {
      const resp = await fetch(`${API_BASE}/api/videos/status/${id}`);
      if (!resp.ok) return;
      const data = await resp.json();
      setStatus(data.status);
      setProgress(data.progress || 0);

      if (data.status === 'completed') {
        setPolling(false);
        await fetchResults(id);
      }

      if (data.status === 'failed') {
        setPolling(false);
        setError('Job failed. Check backend logs for details.');
      }
    } catch (err) {
      console.warn('Status poll failed', err);
    }
  };

  const fetchResults = async (id) => {
    try {
      const resp = await fetch(`${API_BASE}/api/videos/results/${id}`);
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || 'Failed to fetch results');
      }
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch results');
    }
  };

  const renderTopics = () => {
    if (!result?.topics?.length) return null;
    return result.topics.map((topic, idx) => (
      <div className="card" key={`${topic.title}-${idx}`}>
        <div className="card-header">
          <div className="pill">Topic {idx + 1}</div>
          <div className="timestamp">{topic.timestamp_range?.join(' — ')}</div>
        </div>
        <h3>{topic.title}</h3>
        {topic.summary && <p className="muted">{topic.summary}</p>}
        {topic.key_points?.length > 0 && (
          <ul className="bullets">
            {topic.key_points.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        )}

        {/* Phase 4: Visual Sub-topics (Priority) */}
        {topic.sub_topics?.length > 0 ? (
          <div className="frames-grid">
            {topic.sub_topics.map((sub, i) => (
              <a
                className="frame-thumb"
                href={sub.image_url}
                target="_blank"
                rel="noreferrer"
                key={i}
                style={{ 
                  textDecoration: 'none', 
                  padding: 0, 
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                {/* Thumbnail Image */}
                {sub.image_url ? (
                  <div style={{
                    width: '100%',
                    height: '120px',
                    backgroundImage: `url(${sub.image_url})`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    borderBottom: '1px solid rgba(255,255,255,0.06)'
                  }} />
                ) : (
                   <div style={{ 
                     height: '120px', 
                     background: 'linear-gradient(45deg, #1f2937, #111827)', 
                     display: 'flex', 
                     alignItems: 'center', 
                     justifyContent: 'center',
                     color: '#374151',
                     fontSize: '24px'
                   }}>
                     🖼️
                   </div>
                )}
                
                <div style={{ padding: '12px', flex: 1, display: 'flex', flexDirection: 'column' }}>
                  <div className="frame-meta" style={{ marginBottom: '6px' }}>
                    <span className="pill pill-ghost" style={{ fontSize: '10px', padding: '2px 8px' }}>
                      {sub.timestamp}
                    </span>
                  </div>
                  <div className="frame-desc" style={{ padding: 0, marginTop: 0, fontSize: '13px', lineHeight: '1.4' }}>
                    {sub.title}
                  </div>
                  {sub.visual_summary && (
                     <div style={{ marginTop: '6px', fontSize: '11px', color: '#9ca3af', lineHeight: '1.3' }}>
                       {sub.visual_summary.length > 80 ? sub.visual_summary.substring(0, 80) + '...' : sub.visual_summary}
                     </div>
                  )}
                </div>
              </a>
            ))}
          </div>
        ) : (
          /* Fallback to legacy frames */
          topic.frames?.length > 0 && (
            <div className="frames-grid">
              {topic.frames.slice(0, 4).map((f, i) => (
                <a
                  className="frame-thumb"
                  href={f.drive_url}
                  target="_blank"
                  rel="noreferrer"
                  key={i}
                >
                  <div className="frame-meta">
                    <span>{f.timestamp}</span>
                    <span className="pill pill-ghost">{f.type || 'frame'}</span>
                  </div>
                  <div className="frame-desc">{f.description || 'Frame'}</div>
                </a>
              ))}
            </div>
          )
        )}
      </div>
    ));
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>Video Summarizer</h1>
          <p className="muted">
            Upload a video file, or paste a Google Drive or YouTube video link, kick off processing, and see transcript + key frames + insights.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {viewingPastReport && (
            <button
              onClick={() => {
              setViewingPastReport(false);
              setJobId('');
              setResult(null);
              setVideoUrl('');
              setVideoName('');
              navigate('/');
              }}
              style={{
                padding: '10px 20px',
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.2)',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s ease'
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = 'rgba(255,255,255,0.15)';
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = 'rgba(255,255,255,0.1)';
              }}
            >
              ← Back to New Video
            </button>
          )}
          <button
            onClick={() => navigate('/reports')}
            style={{
              padding: '10px 20px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = 'rgba(255,255,255,0.15)';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'rgba(255,255,255,0.1)';
            }}
          >
            📊 Past Reports
          </button>
          <div className="status-chip">
            <span className={`dot dot-${status === 'completed' ? 'green' : status === 'failed' ? 'red' : 'amber'}`} />
            <span>{status === 'idle' || viewingPastReport ? 'Completed' : status}</span>
          </div>
        </div>
      </header>

      {!viewingPastReport && (
      <form className="card form" onSubmit={handleSubmit}>
        <div className="field">
          <label>Video Source</label>
          <select
            value={videoSource}
            onChange={(e) => {
              setVideoSource(e.target.value);
              if (e.target.value !== 'upload') {
                setUploadedFile(null);
              }
              if (e.target.value === 'upload') {
                setVideoUrl('');
              }
            }}
            style={{
              padding: '12px',
              backgroundColor: 'rgba(255,255,255,0.05)',
              color: 'grey',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              fontSize: '14px',
              width: '100%',
              marginBottom: '15px'
            }}
          >
            <option value="auto">Auto-detect from URL</option>
            <option value="upload">Upload Video File</option>
            <option value="drive">Google Drive</option>
            <option value="youtube">YouTube</option>
          </select>
        </div>

        {videoSource === 'upload' ? (
          <div className="field">
            <label>Upload Video File</label>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  setUploadedFile(file);
                  if (!videoName.trim()) {
                    setVideoName(file.name.replace(/\.[^/.]+$/, ''));
                  }
                }
              }}
              required
              style={{
                padding: '12px',
                backgroundColor: 'rgba(255,255,255,0.05)',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.2)',
                borderRadius: '8px',
                fontSize: '14px',
                width: '100%',
                cursor: 'pointer'
              }}
            />
            {uploadedFile && (
              <div style={{ marginTop: '8px', fontSize: '12px', color: '#10b981' }}>
                ✓ Selected: {uploadedFile.name} ({(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB)
              </div>
            )}
          </div>
        ) : (
          <div className="field">
            <label>Video URL (Google Drive or YouTube)</label>
            <input
              type="url"
              placeholder="https://drive.google.com/file/d/FILE_ID/view or https://youtube.com/watch?v=VIDEO_ID"
              value={videoUrl}
              onChange={(e) => {
                setVideoUrl(e.target.value);
                const detected = detectVideoSource(e.target.value);
                if (detected !== 'auto' && videoSource === 'auto') {
                  setVideoSource(detected);
                }
              }}
              required={videoSource !== 'upload'}
            />
            <div style={{ marginTop: '8px', fontSize: '12px', color: '#9ca3af' }}>
              {videoUrl && detectVideoSource(videoUrl) === 'youtube' && '✓ YouTube URL detected'}
              {videoUrl && detectVideoSource(videoUrl) === 'drive' && '✓ Google Drive URL detected'}
              {videoUrl && detectVideoSource(videoUrl) === 'auto' && videoSource !== 'upload' && '⚠️ Please enter a valid Drive or YouTube URL'}
            </div>
          </div>
        )}

        <div className="field">
          <label>Video Name (optional)</label>
          <input
            type="text"
            placeholder="My Seminar / Lecture"
            value={videoName}
            onChange={(e) => setVideoName(e.target.value)}
          />
        </div>

        <div className="actions">
          <button 
            type="submit" 
            disabled={
              (videoSource === 'upload' ? !uploadedFile : !videoUrl.trim()) || 
              status === 'pending' || 
              polling
            }
          >
            {status === 'pending' || polling ? 'Processing…' : 'Start Processing'}
          </button>
          {jobId && !viewingPastReport && <span className="muted">Job ID: {jobId}</span>}
        </div>

        {error && <div className="error">{error}</div>}

        {(status !== 'idle' && status !== 'failed') && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
            <div className="progress-label">{Math.round(progress * 100)}%</div>
          </div>
        )}
      </form>
      )}

      {result && (
        <section className="grid">
          <div className="card">
            <div className="card-header">
              <div className="pill">Summary</div>
              <div className="timestamp">Duration: {result.duration ? `${Math.round(result.duration / 60)} min` : '—'}</div>
            </div>
            {result.video_genre && result.video_genre !== 'unknown' && (
              <div style={{ 
                marginBottom: '15px', 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px' 
              }}>
                <span style={{
                  padding: '4px 12px',
                  backgroundColor: 'rgba(59, 130, 246, 0.2)',
                  color: '#60a5fa',
                  borderRadius: '12px',
                  fontSize: '12px',
                  fontWeight: '500',
                  textTransform: 'capitalize'
                }}>
                  🎬 {result.video_genre.replace(/_/g, ' ')}
                </span>
                {result.genre_confidence && (
                  <span style={{ fontSize: '11px', color: '#9ca3af' }}>
                    ({Math.round(result.genre_confidence * 100)}% confidence)
                  </span>
                )}
              </div>
            )}
            <h3>Executive Summary</h3>
            <p className="muted">{result.executive_summary || 'No summary yet.'}</p>
            {result.key_takeaways?.length > 0 && (
              <div>
                <h4>Key Takeaways</h4>
                <ul className="bullets">
                  {result.key_takeaways.map((k, i) => (
                    <li key={i}>{k}</li>
                  ))}
                </ul>
              </div>
            )}
            {result.entities && (
              <div className="chips">
                {Object.entries(result.entities).flatMap(([type, items]) =>
                  (items || []).map((item, idx) => (
                    <span className="chip" key={`${type}-${idx}`}>
                      {item}
                    </span>
                  ))
                )}
              </div>
            )}
          </div>

          <div className="stack">
            {renderTopics()}
          </div>
        </section>
      )}

      {/* Video Chatbot - only show when results are available */}
      {result && jobId && result.status === 'completed' && (
        <VideoChatBot 
          jobId={jobId} 
          videoName={result.video_name || videoName || 'this video'} 
        />
      )}
    </div>
  );
};

export default Landing;
