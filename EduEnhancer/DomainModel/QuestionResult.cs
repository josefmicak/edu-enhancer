using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class QuestionResult
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int QuestionResultId { get; set; }
        public int TestResultId { get; set; } = default!;
        public int QuestionTemplateId { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public TestResult TestResult { get; set; } = default!;
        public QuestionTemplate QuestionTemplate { get; set; } = default!;
        public ICollection<SubquestionResult> SubquestionResultList { get; set; } = default!;
    }
}