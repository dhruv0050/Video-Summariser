import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import VideoChatBot from '../components/VideoChatBot';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const Landing = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user } = useUser();
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
  const [currentAction, setCurrentAction] = useState('');
  const [logs, setLogs] = useState([]);

  // Detect video source from URL
  const detectVideoSource = (url) => {
    if (!url) return 'auto';
    const youtubePattern = /(?:youtube\.com|youtu\.be)/;
    const drivePattern = /drive\.google\.com/;
    const playlistPattern = /[?&]list=/;

    if (youtubePattern.test(url) && playlistPattern.test(url)) return 'playlist';
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
      interval = setInterval(() => pollStatus(jobId), 8000);
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

    // Handle playlist submission
    if (videoSource === 'playlist') {
      try {
        const resp = await fetch(`${API_BASE}/api/topics/process`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            playlist_url: videoUrl,
            user_id: user?.id || null
          })
        });
        if (!resp.ok) {
          const errData = await resp.json();
          throw new Error(errData.detail || 'Failed to process playlist');
        }
        const data = await resp.json();
        // Navigate to the new topic's detail page
        navigate(`/topics/${data.topic_id}`);
        return;
      } catch (err) {
        setError(err.message || 'Failed to process playlist');
        setStatus('idle');
        return;
      }
    }

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
        if (user?.id) {
          formData.append('user_id', user.id);
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
          user_id: user?.id || null,
        };
      } else {
        endpoint = `${API_BASE}/api/videos/process`;
        payload = {
          drive_video_url: url,
          video_name: videoName.trim() || 'Untitled Video',
          user_id: user?.id || null,
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
      setCurrentAction(data.current_action || '');
      setLogs(data.processing_logs || []);

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

    // Filter out ads/sponsorships from display
    const contentTopics = result.topics.filter(topic => {
      // Check explicit type if available
      if (topic.type === 'ad') return false;

      // Check keywords in title/summary
      const text = (topic.title + ' ' + (topic.summary || '')).toLowerCase();

      // DEBUG: Check what is being filtered
      console.log(`Checking topic: "${topic.title}"`, text);

      if (topic.title.includes("Sponsorship Message")) {
        console.log("Explicitly blocking Sponsorship Message");
        return false;
      }

      const isAd = text.includes('sponsorship') ||
        text.includes('advertisement') ||
        text.includes('promotional message') ||
        text.includes('paid promotion') ||
        text.includes('sponsor'); // Aggressive check for any mention of sponsor

      if (isAd) console.log("Blocked ad:", topic.title);
      return !isAd;
    });

    return contentTopics.map((topic, idx) => (
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
            {topic.sub_topics.map((sub, i) => {
              // DEBUG: Check why image might be missing
              if (!sub.image_url) console.log(`Missing image for subtopic: ${sub.title} at ${sub.timestamp}`, sub);

              return (
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
                      background: 'rgba(255,255,255,0.05)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#9ca3af',
                      gap: '5px',
                      borderBottom: '1px solid rgba(255,255,255,0.06)'
                    }}>
                      <span style={{ fontSize: '20px' }}>🖼️</span>
                      <span style={{ fontSize: '10px' }}>No Preview</span>
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
              );
            })}
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
          <button
            onClick={() => navigate('/topics')}
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
            📚 Topics
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
              <option value="playlist">YouTube Playlist</option>
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
                placeholder={videoSource === 'playlist'
                  ? 'https://www.youtube.com/playlist?list=PLxxxxxx'
                  : 'https://drive.google.com/file/d/FILE_ID/view or https://youtube.com/watch?v=VIDEO_ID'}
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
                {videoUrl && detectVideoSource(videoUrl) === 'playlist' && '✓ YouTube Playlist detected'}
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
              {status === 'pending' || polling ? 'Processing…' : videoSource === 'playlist' ? '📚 Process Playlist' : 'Start Processing'}
            </button>
            {jobId && !viewingPastReport && <span className="muted">Job ID: {jobId}</span>}
          </div>

          {error && <div className="error">{error}</div>}

          {(status !== 'idle' && status !== 'failed') && (
            <div style={{ marginTop: '20px' }}>
              <div className="progress">
                <div className="progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
                <div className="progress-label">{Math.round(progress * 100)}%</div>
              </div>
              
              {/* User-Centric Processing Logs */}
              <div style={{ 
                marginTop: '15px', 
                padding: '15px', 
                backgroundColor: 'rgba(255,255,255,0.03)', 
                borderRadius: '10px',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '10px', 
                  marginBottom: '10px' 
                }}>
                  <div className="loading-spinner-small" style={{ 
                    width: '12px', 
                    height: '12px', 
                    border: '2px solid rgba(59,130,246,0.3)', 
                    borderTopColor: '#3b82f6', 
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }} />
                  <span style={{ fontSize: '14px', fontWeight: '500', color: '#e0e7ff' }}>
                    {currentAction || 'Initializing AI pipeline...'}
                  </span>
                </div>
                
                {logs.length > 0 && (
                  <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    gap: '6px',
                    opacity: 0.7 
                  }}>
                    {logs.slice(-3).reverse().map((log, i) => (
                      <div key={i} style={{ 
                        fontSize: '12px', 
                        color: '#9ca3af',
                        display: 'flex',
                        gap: '8px'
                      }}>
                        <span style={{ color: '#3b82f6' }}>•</span>
                        <span>{log.message}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
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

            {/* 5-Slide Executive Summary */}
            {result.slide_summary?.length > 0 && (
              <div style={{ marginTop: '24px' }}>
                <h4 style={{ marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span>📊 5-Slide Summary</span>
                  <span style={{
                    fontSize: '11px',
                    padding: '3px 10px',
                    borderRadius: '999px',
                    background: 'rgba(99,102,241,0.15)',
                    color: '#a5b4fc',
                    fontWeight: '500',
                    border: '1px solid rgba(99,102,241,0.3)'
                  }}>
                    {result.slide_summary.length} slides
                  </span>
                </h4>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                  gap: '14px',
                }}>
                  {result.slide_summary.map((slide, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: '14px',
                        padding: '18px',
                        transition: 'border-color 0.2s ease, transform 0.2s ease',
                        cursor: 'default',
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(99,102,241,0.4)';
                        e.currentTarget.style.transform = 'translateY(-2px)';
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                        e.currentTarget.style.transform = 'translateY(0)';
                      }}
                    >
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        marginBottom: '12px'
                      }}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: '28px',
                          height: '28px',
                          borderRadius: '8px',
                          background: 'linear-gradient(135deg, rgba(99,102,241,0.3), rgba(34,197,94,0.3))',
                          color: '#e0e7ff',
                          fontSize: '13px',
                          fontWeight: '700',
                          border: '1px solid rgba(99,102,241,0.3)',
                          flexShrink: 0
                        }}>
                          {idx + 1}
                        </span>
                        <span style={{
                          fontSize: '14px',
                          fontWeight: '700',
                          color: '#e0e7ff',
                          lineHeight: '1.3'
                        }}>
                          {slide.title}
                        </span>
                      </div>
                      <ul style={{
                        margin: 0,
                        paddingLeft: '16px',
                        listStyle: 'none',
                      }}>
                        {slide.bullets?.map((bullet, bIdx) => (
                          <li
                            key={bIdx}
                            style={{
                              fontSize: '13px',
                              color: '#c3c9e6',
                              lineHeight: '1.5',
                              marginBottom: '6px',
                              position: 'relative',
                              paddingLeft: '14px',
                            }}
                          >
                            <span style={{
                              position: 'absolute',
                              left: 0,
                              top: '6px',
                              width: '5px',
                              height: '5px',
                              borderRadius: '50%',
                              background: 'rgba(99,102,241,0.6)',
                            }} />
                            <span dangerouslySetInnerHTML={{
                              __html: bullet.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#e0e7ff;font-weight:600">$1</strong>')
                            }} />
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
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
