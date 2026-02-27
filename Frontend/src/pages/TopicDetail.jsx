import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const TopicDetail = () => {
    const { topicId } = useParams();
    const navigate = useNavigate();
    const { user } = useUser();
    const [topic, setTopic] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [selectedVideoIndex, setSelectedVideoIndex] = useState(0);
    const [videoResult, setVideoResult] = useState(null);
    const [loadingReport, setLoadingReport] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);

    const fetchTopic = async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/topics/${topicId}`);
            if (!resp.ok) throw new Error('Topic not found');
            const data = await resp.json();
            setTopic(data);

            // Auto-select first completed video
            const firstCompleted = data.videos?.findIndex(v => v.status === 'completed');
            if (firstCompleted >= 0) {
                setSelectedVideoIndex(firstCompleted);
                loadVideoReport(data.videos[firstCompleted].job_id);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const loadVideoReport = async (jobId) => {
        if (!jobId) return;
        setLoadingReport(true);
        try {
            const resp = await fetch(`${API_BASE}/api/videos/results/${jobId}`);
            if (!resp.ok) throw new Error('Report not available');
            const data = await resp.json();
            setVideoResult(data);
        } catch (err) {
            setVideoResult(null);
        } finally {
            setLoadingReport(false);
        }
    };

    useEffect(() => {
        fetchTopic();
    }, [topicId]);

    // Poll for progress when processing
    useEffect(() => {
        if (!topic || topic.status !== 'processing') return;
        const interval = setInterval(async () => {
            try {
                const resp = await fetch(`${API_BASE}/api/topics/${topicId}/progress`);
                const progress = await resp.json();
                setTopic(prev => prev ? { ...prev, ...progress } : prev);
                // Refresh full topic when progress changes
                if (progress.status === 'completed') {
                    clearInterval(interval);
                    fetchTopic();
                }
            } catch (e) { /* skip */ }
        }, 8000);
        return () => clearInterval(interval);
    }, [topic?.status]);

    const selectVideo = (index) => {
        setSelectedVideoIndex(index);
        const video = topic?.videos?.[index];
        if (video?.job_id && video.status === 'completed') {
            loadVideoReport(video.job_id);
        } else {
            setVideoResult(null);
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed': return '✅';
            case 'processing': return '⏳';
            case 'failed': return '❌';
            default: return '⬜';
        }
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', color: '#9ca3af' }}>
                Loading topic...
            </div>
        );
    }

    if (error) {
        return (
            <div style={{ maxWidth: '600px', margin: '80px auto', textAlign: 'center' }}>
                <p style={{ color: '#fca5a5' }}>{error}</p>
                <button onClick={() => navigate('/topics')} style={{ marginTop: '16px', padding: '10px 20px', background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', cursor: 'pointer' }}>
                    ← Back to Topics
                </button>
            </div>
        );
    }

    if (!topic) return null;

    const completedCount = topic.videos?.filter(v => v.status === 'completed').length || 0;

    return (
        <div style={{ display: 'flex', height: 'calc(100vh - 60px)', overflow: 'hidden' }}>
            {/* Sidebar - Table of Contents */}
            <div style={{
                width: sidebarOpen ? '300px' : '0px',
                minWidth: sidebarOpen ? '300px' : '0px',
                borderRight: '1px solid rgba(255,255,255,0.06)',
                background: 'rgba(0,0,0,0.2)',
                display: 'flex',
                flexDirection: 'column',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
            }}>
                {/* Topic Header */}
                <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <button
                        onClick={() => navigate('/topics')}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: '#818cf8',
                            cursor: 'pointer',
                            fontSize: '12px',
                            padding: 0,
                            marginBottom: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                        }}
                    >
                        ← Back to Topics
                    </button>
                    <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#e0e7ff', margin: '0 0 8px 0', lineHeight: '1.3' }}>
                        {topic.title}
                    </h2>
                    {topic.channel && (
                        <p style={{ fontSize: '12px', color: '#6b7280', margin: '0 0 12px 0' }}>{topic.channel}</p>
                    )}
                    {/* Progress */}
                    <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '6px' }}>
                        {completedCount} of {topic.video_count} videos processed
                    </div>
                    <div style={{
                        height: '3px',
                        backgroundColor: 'rgba(255,255,255,0.06)',
                        borderRadius: '999px',
                        overflow: 'hidden'
                    }}>
                        <div style={{
                            height: '100%',
                            width: `${(topic.progress || 0) * 100}%`,
                            background: topic.status === 'completed'
                                ? 'linear-gradient(90deg, #22c55e, #10b981)'
                                : 'linear-gradient(90deg, #6366f1, #818cf8)',
                            borderRadius: '999px',
                            transition: 'width 0.5s ease'
                        }} />
                    </div>
                </div>

                {/* Chapter List */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
                    {topic.videos?.map((video, idx) => (
                        <button
                            key={idx}
                            onClick={() => selectVideo(idx)}
                            style={{
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: '10px',
                                width: '100%',
                                padding: '12px',
                                background: selectedVideoIndex === idx ? 'rgba(99,102,241,0.15)' : 'transparent',
                                border: selectedVideoIndex === idx ? '1px solid rgba(99,102,241,0.3)' : '1px solid transparent',
                                borderRadius: '10px',
                                cursor: video.status === 'completed' ? 'pointer' : 'default',
                                textAlign: 'left',
                                color: 'inherit',
                                transition: 'all 0.15s ease',
                                marginBottom: '2px',
                                opacity: video.status === 'pending' ? 0.5 : 1
                            }}
                        >
                            <span style={{
                                fontSize: '12px',
                                minWidth: '22px',
                                height: '22px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                borderRadius: '6px',
                                background: selectedVideoIndex === idx ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.05)',
                                color: '#e0e7ff',
                                fontWeight: '700',
                                flexShrink: 0
                            }}>
                                {idx + 1}
                            </span>
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{
                                    fontSize: '13px',
                                    fontWeight: selectedVideoIndex === idx ? '600' : '500',
                                    color: selectedVideoIndex === idx ? '#e0e7ff' : '#c3c9e6',
                                    lineHeight: '1.3',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    display: '-webkit-box',
                                    WebkitLineClamp: 2,
                                    WebkitBoxOrient: 'vertical'
                                }}>
                                    {video.video_title}
                                </div>
                            </div>
                            <span style={{ fontSize: '14px', flexShrink: 0 }}>
                                {getStatusIcon(video.status)}
                            </span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Toggle sidebar button */}
            <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                style={{
                    position: 'absolute',
                    left: sidebarOpen ? '300px' : '0px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: '24px',
                    height: '48px',
                    background: 'rgba(99,102,241,0.2)',
                    border: '1px solid rgba(99,102,241,0.3)',
                    borderLeft: 'none',
                    borderRadius: '0 8px 8px 0',
                    color: '#818cf8',
                    cursor: 'pointer',
                    transition: 'left 0.3s ease',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '12px',
                    zIndex: 10
                }}
            >
                {sidebarOpen ? '‹' : '›'}
            </button>

            {/* Main Content - Report Viewer */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '32px 40px' }}>
                {/* Topic Summary (if available) */}
                {topic.topic_summary && selectedVideoIndex === 0 && (
                    <div style={{
                        background: 'rgba(99,102,241,0.08)',
                        border: '1px solid rgba(99,102,241,0.2)',
                        borderRadius: '14px',
                        padding: '20px',
                        marginBottom: '24px'
                    }}>
                        <h4 style={{ color: '#a5b4fc', margin: '0 0 8px 0', fontSize: '14px' }}>📖 Topic Overview</h4>
                        <p style={{ color: '#c3c9e6', fontSize: '14px', lineHeight: '1.6', margin: 0 }}>{topic.topic_summary}</p>
                        {topic.learning_objectives?.length > 0 && (
                            <div style={{ marginTop: '12px' }}>
                                <h5 style={{ color: '#a5b4fc', margin: '0 0 6px 0', fontSize: '12px' }}>Learning Objectives</h5>
                                <ul style={{ margin: 0, paddingLeft: '16px' }}>
                                    {topic.learning_objectives.map((obj, i) => (
                                        <li key={i} style={{ color: '#c3c9e6', fontSize: '13px', marginBottom: '4px' }}>{obj}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}

                {/* Selected Video Report */}
                {topic.videos?.[selectedVideoIndex] && (
                    <>
                        {/* Chapter Header */}
                        <div style={{ marginBottom: '24px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                                <span style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    width: '32px',
                                    height: '32px',
                                    borderRadius: '10px',
                                    background: 'linear-gradient(135deg, rgba(99,102,241,0.3), rgba(34,197,94,0.3))',
                                    color: '#e0e7ff',
                                    fontSize: '14px',
                                    fontWeight: '700',
                                    border: '1px solid rgba(99,102,241,0.3)'
                                }}>
                                    {selectedVideoIndex + 1}
                                </span>
                                <h2 style={{ fontSize: '22px', fontWeight: '700', color: '#e0e7ff', margin: 0 }}>
                                    {topic.videos[selectedVideoIndex].video_title}
                                </h2>
                            </div>
                            <div style={{ fontSize: '12px', color: '#6b7280' }}>
                                Chapter {selectedVideoIndex + 1} of {topic.video_count}
                                {topic.videos[selectedVideoIndex].duration > 0 && (
                                    <span> · {Math.round(topic.videos[selectedVideoIndex].duration / 60)} min</span>
                                )}
                            </div>
                        </div>

                        {/* Report Content */}
                        {loadingReport ? (
                            <div style={{ textAlign: 'center', padding: '60px', color: '#9ca3af' }}>
                                Loading report...
                            </div>
                        ) : topic.videos[selectedVideoIndex].status === 'completed' && videoResult ? (
                            <div>
                                {/* Executive Summary */}
                                <div className="card" style={{ marginBottom: '20px' }}>
                                    <h3>Executive Summary</h3>
                                    <p className="muted">{videoResult.executive_summary || 'No summary available.'}</p>
                                </div>

                                {/* Key Takeaways */}
                                {videoResult.key_takeaways?.length > 0 && (
                                    <div className="card" style={{ marginBottom: '20px' }}>
                                        <h4>Key Takeaways</h4>
                                        <ul className="bullets">
                                            {videoResult.key_takeaways.map((k, i) => (
                                                <li key={i}>{k}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* 5-Slide Summary */}
                                {videoResult.slide_summary?.length > 0 && (
                                    <div style={{ marginBottom: '20px' }}>
                                        <h4 style={{ marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span>📊 5-Slide Summary</span>
                                            <span style={{
                                                fontSize: '11px', padding: '3px 10px', borderRadius: '999px',
                                                background: 'rgba(99,102,241,0.15)', color: '#a5b4fc', fontWeight: '500',
                                                border: '1px solid rgba(99,102,241,0.3)'
                                            }}>
                                                {videoResult.slide_summary.length} slides
                                            </span>
                                        </h4>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '12px' }}>
                                            {videoResult.slide_summary.map((slide, idx) => (
                                                <div key={idx} style={{
                                                    background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
                                                    borderRadius: '12px', padding: '16px'
                                                }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                                                        <span style={{
                                                            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                                                            width: '24px', height: '24px', borderRadius: '6px',
                                                            background: 'linear-gradient(135deg, rgba(99,102,241,0.3), rgba(34,197,94,0.3))',
                                                            color: '#e0e7ff', fontSize: '12px', fontWeight: '700'
                                                        }}>
                                                            {idx + 1}
                                                        </span>
                                                        <span style={{ fontSize: '13px', fontWeight: '700', color: '#e0e7ff' }}>
                                                            {slide.title}
                                                        </span>
                                                    </div>
                                                    <ul style={{ margin: 0, paddingLeft: '14px', listStyle: 'none' }}>
                                                        {slide.bullets?.map((bullet, bIdx) => (
                                                            <li key={bIdx} style={{
                                                                fontSize: '12px', color: '#c3c9e6', lineHeight: '1.5', marginBottom: '4px',
                                                                position: 'relative', paddingLeft: '12px',
                                                            }}>
                                                                <span style={{
                                                                    position: 'absolute', left: 0, top: '6px', width: '4px', height: '4px',
                                                                    borderRadius: '50%', background: 'rgba(99,102,241,0.6)'
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

                                {/* Topics */}
                                {videoResult.topics?.length > 0 && (
                                    <div style={{ marginBottom: '20px' }}>
                                        <h4 style={{ marginBottom: '14px' }}>Topics Covered</h4>
                                        {videoResult.topics.filter(t => t.type !== 'ad' && !t.title?.toLowerCase().includes('sponsor')).map((topicItem, idx) => (
                                            <div className="card" key={idx} style={{ marginBottom: '10px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                                    <span className="pill">Topic {idx + 1}</span>
                                                    <span style={{ fontSize: '11px', color: '#6b7280' }}>{topicItem.timestamp_range?.join(' — ')}</span>
                                                </div>
                                                <h4 style={{ margin: '4px 0 6px 0' }}>{topicItem.title}</h4>
                                                {topicItem.summary && <p className="muted" style={{ fontSize: '13px' }}>{topicItem.summary}</p>}

                                                {/* Visual Frames for this topic */}
                                                {topicItem.frames?.length > 0 && (
                                                    <div style={{
                                                        display: 'grid',
                                                        gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                                                        gap: '8px',
                                                        marginTop: '10px'
                                                    }}>
                                                        {topicItem.frames.map((frame, fIdx) => (
                                                            <a
                                                                key={fIdx}
                                                                href={frame.drive_url || frame.image_url}
                                                                target="_blank"
                                                                rel="noreferrer"
                                                                style={{
                                                                    display: 'block',
                                                                    borderRadius: '8px',
                                                                    overflow: 'hidden',
                                                                    border: '1px solid rgba(255,255,255,0.08)',
                                                                    textDecoration: 'none',
                                                                    transition: 'border-color 0.2s ease'
                                                                }}
                                                            >
                                                                {(frame.image_url || frame.drive_url) ? (
                                                                    <div style={{
                                                                        width: '100%',
                                                                        height: '90px',
                                                                        backgroundImage: `url(${frame.image_url || frame.drive_url})`,
                                                                        backgroundSize: 'cover',
                                                                        backgroundPosition: 'center'
                                                                    }} />
                                                                ) : (
                                                                    <div style={{
                                                                        width: '100%',
                                                                        height: '90px',
                                                                        background: 'rgba(255,255,255,0.05)',
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        justifyContent: 'center',
                                                                        color: '#6b7280',
                                                                        fontSize: '20px'
                                                                    }}>🖼️</div>
                                                                )}
                                                                <div style={{ padding: '6px 8px' }}>
                                                                    {frame.timestamp && (
                                                                        <span style={{
                                                                            fontSize: '10px',
                                                                            color: '#818cf8',
                                                                            background: 'rgba(99,102,241,0.15)',
                                                                            padding: '1px 6px',
                                                                            borderRadius: '4px'
                                                                        }}>{frame.timestamp}</span>
                                                                    )}
                                                                    {frame.description && (
                                                                        <div style={{
                                                                            fontSize: '11px',
                                                                            color: '#9ca3af',
                                                                            marginTop: '4px',
                                                                            overflow: 'hidden',
                                                                            textOverflow: 'ellipsis',
                                                                            whiteSpace: 'nowrap'
                                                                        }}>{frame.description}</div>
                                                                    )}
                                                                </div>
                                                            </a>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Entities */}
                                {videoResult.entities && Object.keys(videoResult.entities).length > 0 && (
                                    <div className="chips" style={{ marginBottom: '20px' }}>
                                        {Object.entries(videoResult.entities).flatMap(([type, items]) =>
                                            (items || []).map((item, idx) => (
                                                <span className="chip" key={`${type}-${idx}`}>{item}</span>
                                            ))
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : topic.videos[selectedVideoIndex].status === 'processing' ? (
                            <div style={{ textAlign: 'center', padding: '60px' }}>
                                <div style={{ fontSize: '40px', marginBottom: '16px', animation: 'pulse 2s infinite' }}>⏳</div>
                                <p style={{ color: '#f59e0b', fontSize: '16px', fontWeight: '600' }}>Processing this video...</p>
                                <p style={{ color: '#9ca3af', fontSize: '13px' }}>The report will appear here when ready</p>
                            </div>
                        ) : topic.videos[selectedVideoIndex].status === 'failed' ? (
                            <div style={{ textAlign: 'center', padding: '60px' }}>
                                <div style={{ fontSize: '40px', marginBottom: '16px' }}>❌</div>
                                <p style={{ color: '#ef4444', fontSize: '16px', fontWeight: '600' }}>Processing failed</p>
                                <p style={{ color: '#9ca3af', fontSize: '13px' }}>This video encountered an error during processing</p>
                            </div>
                        ) : topic.videos[selectedVideoIndex].status === 'completed' && !videoResult ? (
                            <div style={{ textAlign: 'center', padding: '60px' }}>
                                <div style={{ fontSize: '40px', marginBottom: '16px' }}>⚠️</div>
                                <p style={{ color: '#f59e0b', fontSize: '16px', fontWeight: '600' }}>Report unavailable</p>
                                <p style={{ color: '#9ca3af', fontSize: '13px' }}>The video was processed but the report could not be loaded</p>
                            </div>
                        ) : (
                            <div style={{ textAlign: 'center', padding: '60px' }}>
                                <div style={{ fontSize: '40px', marginBottom: '16px' }}>⬜</div>
                                <p style={{ color: '#6b7280', fontSize: '16px' }}>Waiting to be processed</p>
                                <p style={{ color: '#4b5563', fontSize: '13px' }}>This video is queued and will be processed in order</p>
                            </div>
                        )}

                        {/* Navigation */}
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginTop: '32px',
                            paddingTop: '20px',
                            borderTop: '1px solid rgba(255,255,255,0.06)'
                        }}>
                            <button
                                onClick={() => selectVideo(selectedVideoIndex - 1)}
                                disabled={selectedVideoIndex === 0}
                                style={{
                                    padding: '10px 20px',
                                    background: selectedVideoIndex === 0 ? 'rgba(255,255,255,0.03)' : 'rgba(99,102,241,0.15)',
                                    border: `1px solid ${selectedVideoIndex === 0 ? 'rgba(255,255,255,0.06)' : 'rgba(99,102,241,0.3)'}`,
                                    borderRadius: '10px',
                                    color: selectedVideoIndex === 0 ? '#4b5563' : '#a5b4fc',
                                    cursor: selectedVideoIndex === 0 ? 'default' : 'pointer',
                                    fontSize: '13px',
                                    fontWeight: '600'
                                }}
                            >
                                ← Previous Chapter
                            </button>
                            <button
                                onClick={() => selectVideo(selectedVideoIndex + 1)}
                                disabled={selectedVideoIndex >= (topic.videos?.length || 1) - 1}
                                style={{
                                    padding: '10px 20px',
                                    background: selectedVideoIndex >= (topic.videos?.length || 1) - 1 ? 'rgba(255,255,255,0.03)' : 'rgba(99,102,241,0.15)',
                                    border: `1px solid ${selectedVideoIndex >= (topic.videos?.length || 1) - 1 ? 'rgba(255,255,255,0.06)' : 'rgba(99,102,241,0.3)'}`,
                                    borderRadius: '10px',
                                    color: selectedVideoIndex >= (topic.videos?.length || 1) - 1 ? '#4b5563' : '#a5b4fc',
                                    cursor: selectedVideoIndex >= (topic.videos?.length || 1) - 1 ? 'default' : 'pointer',
                                    fontSize: '13px',
                                    fontWeight: '600'
                                }}
                            >
                                Next Chapter →
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default TopicDetail;
