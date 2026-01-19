import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const Reports = () => {
  const navigate = useNavigate();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const [limit] = useState(10);
  const [status, setStatus] = useState('');
  const [totalReports, setTotalReports] = useState(0);

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const fetchReports = async () => {
    setLoading(true);
    setError('');
    try {
      const query = new URLSearchParams();
      query.append('page', page);
      query.append('limit', limit);
      if (status) query.append('status', status);

      const response = await fetch(`${API_BASE}/api/videos/reports?${query}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch reports');
      }

      const data = await response.json();
      setReports(data);
      setTotalReports(data.length);
    } catch (err) {
      setError(err.message);
      setReports([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [page, status]);

  const handleViewReport = (jobId) => {
    navigate(`/?jobId=${jobId}`);
  };

  const getStatusBadge = (status) => {
    const statusStyles = {
      completed: { bg: '#10b981', text: '✓ Completed' },
      processing: { bg: '#f59e0b', text: '⟳ Processing' },
      pending: { bg: '#6b7280', text: '○ Pending' },
      failed: { bg: '#ef4444', text: '✕ Failed' }
    };
    
    const style = statusStyles[status] || statusStyles.pending;
    return (
      <span style={{
        display: 'inline-block',
        backgroundColor: style.bg,
        color: 'white',
        padding: '4px 12px',
        borderRadius: '12px',
        fontSize: '12px',
        fontWeight: '500'
      }}>
        {style.text}
      </span>
    );
  };

  const getVideoLinks = (report) => {
    // Prefer explicit YouTube id/url
    if (report.youtube_video_id) {
      return {
        embed: `https://www.youtube.com/embed/${report.youtube_video_id}`,
        link: report.youtube_url || `https://youtu.be/${report.youtube_video_id}`
      };
    }
    // Handle Drive file id or URL
    if (report.drive_file_id) {
      return {
        embed: `https://drive.google.com/file/d/${report.drive_file_id}/preview`,
        link: `https://drive.google.com/file/d/${report.drive_file_id}/view?usp=sharing`
      };
    }
    if (report.drive_video_url) {
      // Derive id if possible
      const match = report.drive_video_url.match(/file\/d\/([^/]+)/);
      const fileId = match ? match[1] : null;
      return {
        embed: fileId
          ? `https://drive.google.com/file/d/${fileId}/preview`
          : report.drive_video_url,
        link: report.drive_video_url
      };
    }
    return { embed: null, link: null };
  };

  return (
    <div className="page">
      <header style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ margin: 0, fontSize: '28px' }}>📊 Past Reports</h1>
          <button
            onClick={() => navigate('/')}
            style={{
              padding: '10px 20px',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            ← New Video
          </button>
        </div>
      </header>

      <div style={{ padding: '20px' }}>
        {/* Filters */}
        <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
          <select
            value={status}
            onChange={(e) => {
              setStatus(e.target.value);
              setPage(1);
            }}
            style={{
              padding: '10px 15px',
              backgroundColor: 'rgba(255,255,255,0.05)',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              fontSize: '14px'
            }}
          >
            <option value="">All Status</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="pending">Pending</option>
            <option value="failed">Failed</option>
          </select>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <div style={{ fontSize: '14px', color: '#9ca3af' }}>Loading reports...</div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div style={{
            padding: '15px',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderLeft: '3px solid #ef4444',
            borderRadius: '6px',
            color: '#fca5a5',
            marginBottom: '20px'
          }}>
            Error: {error}
          </div>
        )}

        {/* Reports Grid */}
        {!loading && reports.length > 0 ? (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
            gap: '20px'
          }}>
            {reports.map((report) => (
              <div
                key={report.job_id}
                className="card"
                style={{
                  padding: '20px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px',
                  backgroundColor: 'rgba(255,255,255,0.02)',
                  backdropFilter: 'blur(10px)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '15px'
                }}
                onClick={() => handleViewReport(report.job_id)}
              >
                {/* Video Embed */}
                {(() => {
                  const links = getVideoLinks(report);
                  return (
                    <div style={{
                      position: 'relative',
                      height: '0',
                      paddingBottom: '56.25%', // 16:9
                      borderRadius: '10px',
                      overflow: 'hidden',
                      background: 'linear-gradient(135deg, rgba(59,130,246,0.25), rgba(16,185,129,0.25))',
                      border: '1px solid rgba(255,255,255,0.05)'
                    }}>
                      {links.embed ? (
                        <iframe
                          src={links.embed}
                          title={report.video_name || 'Video preview'}
                          style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            border: '0'
                          }}
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                          allowFullScreen
                        />
                      ) : (
                        <div style={{
                          position: 'absolute',
                          inset: 0,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#d1d5db',
                          fontSize: '32px',
                          background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(16,185,129,0.15))'
                        }}>
                          🎬
                        </div>
                      )}
                      <div style={{
                        position: 'absolute',
                        top: '10px',
                        right: '10px'
                      }}>
                        {getStatusBadge(report.status)}
                      </div>
                      {links.link && (
                        <a
                          href={links.link}
                          target="_blank"
                          rel="noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          style={{
                            position: 'absolute',
                            left: '10px',
                            bottom: '10px',
                            padding: '8px 12px',
                            backgroundColor: 'rgba(0,0,0,0.55)',
                            color: 'white',
                            borderRadius: '6px',
                            fontSize: '12px',
                            textDecoration: 'none',
                            border: '1px solid rgba(255,255,255,0.2)'
                          }}
                        >
                          ▶ Open video
                        </a>
                      )}
                    </div>
                  );
                })()}

                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                  <div style={{ flex: 1 }}>
                    <h3 style={{
                      margin: '0 0 5px 0',
                      fontSize: '18px',
                      color: '#fff',
                      wordBreak: 'break-word'
                    }}>
                      {report.video_name || 'Untitled Video'}
                    </h3>
                    <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '5px' }}>
                      ID: {report.job_id.substring(0, 12)}...
                    </div>
                    {report.video_genre && report.video_genre !== 'unknown' && (
                      <div style={{ marginTop: '5px' }}>
                        <span style={{
                          padding: '3px 10px',
                          backgroundColor: 'rgba(59, 130, 246, 0.2)',
                          color: '#60a5fa',
                          borderRadius: '10px',
                          fontSize: '11px',
                          fontWeight: '500',
                          textTransform: 'capitalize'
                        }}>
                          🎬 {report.video_genre.replace(/_/g, ' ')}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Summary */}
                {report.executive_summary && (
                  <p style={{
                    margin: '0',
                    fontSize: '13px',
                    color: '#d1d5db',
                    lineHeight: '1.5',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical'
                  }}>
                    {report.executive_summary}
                  </p>
                )}

                {/* Stats */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '10px',
                  paddingTop: '10px',
                  borderTop: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <div>
                    <div style={{ fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase' }}>Topics</div>
                    <div style={{ fontSize: '20px', fontWeight: '600', color: '#3b82f6' }}>
                      {report.topics_count || 0}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase' }}>Duration</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#10b981' }}>
                      {formatDuration(report.duration)}
                    </div>
                  </div>
                </div>

                {/* Date */}
                <div style={{
                  fontSize: '12px',
                  color: '#6b7280',
                  paddingTop: '10px',
                  borderTop: '1px solid rgba(255,255,255,0.05)'
                }}>
                  Created: {formatDate(report.created_at)}
                  {report.completed_at && (
                    <>
                      <br />
                      Completed: {formatDate(report.completed_at)}
                    </>
                  )}
                </div>

                {/* CTA Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleViewReport(report.job_id);
                  }}
                  style={{
                    padding: '10px',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    color: '#3b82f6',
                    border: '1px solid #3b82f6',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseOver={(e) => {
                    e.target.style.backgroundColor = '#3b82f6';
                    e.target.style.color = 'white';
                  }}
                  onMouseOut={(e) => {
                    e.target.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
                    e.target.style.color = 'black';
                  }}
                >
                  View Report →
                </button>
              </div>
            ))}
          </div>
        ) : (
          !loading && (
            <div style={{
              textAlign: 'center',
              padding: '60px 20px',
              color: '#9ca3af'
            }}>
              <div style={{ fontSize: '48px', marginBottom: '15px' }}>📭</div>
              <div style={{ fontSize: '16px' }}>No reports found</div>
              <div style={{ fontSize: '14px', marginTop: '10px' }}>
                {status ? 'Try changing the filter or ' : ''}
                <span
                  onClick={() => navigate('/')}
                  style={{
                    color: '#3b82f6',
                    cursor: 'pointer',
                    textDecoration: 'underline'
                  }}
                >
                  process a new video
                </span>
              </div>
            </div>
          )
        )}

        {/* Pagination */}
        {reports.length > 0 && (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '10px',
            marginTop: '40px',
            paddingTop: '20px',
            borderTop: '1px solid rgba(255,255,255,0.1)'
          }}>
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              style={{
                padding: '8px 15px',
                backgroundColor: page === 1 ? 'rgba(255,255,255,0.05)' : 'rgba(59, 130, 246, 0.1)',
                color: page === 1 ? '#6b7280' : '#3b82f6',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '6px',
                cursor: page === 1 ? 'not-allowed' : 'pointer',
                fontSize: '13px'
              }}
            >
              ← Previous
            </button>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              color: '#9ca3af',
              fontSize: '13px'
            }}>
              Page {page}
            </div>
            <button
              onClick={() => setPage(page + 1)}
              disabled={reports.length < limit}
              style={{
                padding: '8px 15px',
                backgroundColor: reports.length < limit ? 'rgba(255,255,255,0.05)' : 'rgba(59, 130, 246, 0.1)',
                color: reports.length < limit ? '#6b7280' : '#3b82f6',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '6px',
                cursor: reports.length < limit ? 'not-allowed' : 'pointer',
                fontSize: '13px'
              }}
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Reports;
