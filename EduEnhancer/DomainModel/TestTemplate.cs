using Common;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class TestTemplate
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int TestTemplateId { get; set; }
        public string Title { get; set; } = default!;
        public EnumTypes.NegativePoints NegativePoints { get; set; } = EnumTypes.NegativePoints.Disabled;
        public double? MinimumPoints { get; set; }
        public Subject Subject { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public User Owner { get; set; } = default!;
        public ICollection<QuestionTemplate> QuestionTemplates { get; set; } = default!;
        public bool IsTestingData { get; set; } = false;
    }
}