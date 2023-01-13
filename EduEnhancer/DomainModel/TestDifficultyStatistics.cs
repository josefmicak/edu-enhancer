using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DomainModel
{
    public class TestDifficultyStatistics
    {
        [Key]
        public string UserLogin { get; set; } = default!;
        public User User { get; set; } = default!;
        public string SubquestionTypeAveragePoints { get; set; }
        [NotMapped]
        public double[] InternalSubquestionTypeAveragePoints
        {
            get
            {
                return Array.ConvertAll(SubquestionTypeAveragePoints.Split('|'), Double.Parse);
            }
            set
            {
                double[] _data = value;
                SubquestionTypeAveragePoints = String.Join("|", _data.Select(p => p.ToString()).ToArray());
            }
        }
        public string SubjectIds { get; set; }
        [NotMapped]
        public double[]? InternalSubjectIds
        {
            get
            {
                return Array.ConvertAll(SubjectIds.Split('|'), Double.Parse);
            }
            set
            {
                double[] _data = value;
                SubjectIds = String.Join("|", _data.Select(p => p.ToString()).ToArray());
            }
        }
        public string SubjectAveragePoints { get; set; }
        [NotMapped]
        public double[]? InternalSubjectAveragePoints
        {
            get
            {
                return Array.ConvertAll(SubjectAveragePoints.Split('|'), Double.Parse);
            }
            set
            {
                double[] _data = value;
                SubjectAveragePoints = String.Join("|", _data.Select(p => p.ToString()).ToArray());
            }
        }
        public string SubquestionTypeAverageAnswerCorrectness { get; set; }
        [NotMapped]
        public double[]? InternalSubquestionTypeAverageAnswerCorrectness
        {
            get
            {
                return Array.ConvertAll(SubquestionTypeAverageAnswerCorrectness.Split('|'), Double.Parse);
            }
            set
            {
                double[] _data = value;
                SubquestionTypeAverageAnswerCorrectness = String.Join("|", _data.Select(p => p.ToString()).ToArray());
            }
        }
    }
}
